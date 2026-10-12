from typing import Optional
import numpy as np
import math


class RobotArm:
    def __init__(self) -> None:
        self.links = []
        self.fixed_links = set()

    def add_revolve_link(self, theta: float, alpha: float, a: float, d: float):
        """
        Adds the link with the given parameters to kinematic structure of the leg.
        parameters:
            theta, alpha, a, d: Denavit-Hartenberg pararemters
        """
        # create dh-matrix
        def dh_matrix(phi):
            cos = math.cos(theta + phi)
            sin = math.sin(theta + phi)
            cos_a = math.cos(alpha)
            sin_a = math.sin(alpha)
            return np.array([[cos, -sin * cos_a, sin * sin_a, a * cos],
                             [sin, cos * cos_a, -cos * sin_a, a * sin],
                             [0, sin_a, cos_a, d],
                             [0, 0, 0, 1]])
        self.links.append(dh_matrix)

    def add_fixed_link(self, theta: float, alpha: float, a: float, d: float):
        """
        Adds the link with the given parameters to kinematic structure of the leg.
        parameters:
            theta, alpha, a, d: Denavit-Hartenberg pararemters
        """
        # create dh-matrix
        cos = math.cos(theta)
        sin = math.sin(theta)
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        transform = np.array([[cos, -sin * cos_a, sin * sin_a, a * cos],
                              [sin, cos * cos_a, -cos * sin_a, a * sin],
                              [0, sin_a, cos_a, d],
                              [0, 0, 0, 1]])
        self.links.append((lambda T: lambda _: T)(transform))
        self.fixed_links.add(len(self.links) - 1)

    def _jacobian(self, dh_matrices):
        """
        dh_matrices: list of 4x4 transforms T_0i (frame 0 to frame i)
                     First entry T_00 = Identity (base frame)
        Returns: 6xn Jacobian
        """
        n = len(dh_matrices) - 1  # number of joints
        T_0n = dh_matrices[-1]
        p_e = T_0n[:3, 3]

        # Only compute columns for actuated joints
        actuated = [i for i in range(n) if i not in self.fixed_links]
        J = np.zeros((6, len(actuated)))

        for col, i in enumerate(actuated):
            T_0i = dh_matrices[i]
            z_i = T_0i[:3, 2]
            p_i = T_0i[:3, 3]
            J[:3, col] = np.cross(z_i, p_e - p_i)
            J[3:, col] = z_i

        return J

    def _expand_angles(self, angles: np.ndarray) -> np.ndarray:
        """wrapper function to expand the user given angles to an usable format. Fixed links are assigned an zero angle."""
        actuated = np.array([i for i in range(len(self.links)) if i not in self.fixed_links])

        assert len(angles) == len(
            actuated),  "There must be the same number of angles as revolut joints"

        expanded = np.zeros(len(self.links))
        expanded[actuated] = angles
        return expanded

    def _forward_kin(self, phi: np.ndarray) -> np.ndarray:
        """
        calculates the position of the endeffector.

        returns:
            T: 4x4 homogenous array
        """
        assert len(phi) == len(self.links) - \
            len(self.fixed_links),  "There must be the same number of angles as revolut joints"
        phi_ex = self._expand_angles(phi)
        T = np.eye(4)
        for link, p in zip(self.links, phi_ex):
            T = T @ link(p)
        return T

    def _forward_kin_chain(self, phi: np.ndarray) -> list[np.ndarray]:
        """
        calculates the position for each joint.

        returns:
            T: nx4x4 list of homogenous arrays
        """
        assert len(phi) == len(self.links) - \
            len(self.fixed_links), "There must be the same number of angles as revolut joints"
        phi_ex = self._expand_angles(phi)
        T = []
        T.append(np.eye(4))
        for link, p in zip(self.links, phi_ex):
            T.append(T[-1] @ link(p))
        return T

    def _inverse_kin(self, endeffector_pos: np.ndarray, curr_angle: Optional[np.ndarray] = None) -> np.ndarray:
        """
        calculates the angles for each joint that are needed to reach an given position.

        parameters:
            endeffector_pos: The position the endeffector should approach. Given in the base koordinate system
            curr_angle: [Optional] The current angle of each joint. Uses this as its starting point
        returns:
            angles: array with the angles for the joints in [-pi,pi] format
        """
        max_iter = 50
        damping = 0.01
        max_step = 0.1
        tol = 1e-4

        actuated = np.array([i for i in range(len(self.links)) if i not in self.fixed_links])

        if curr_angle is None:
            curr_angle = np.zeros(len(self.links))
        else:
            curr_angle = self._expand_angles(curr_angle)

        for i in range(max_iter):
            T_curr = self._forward_kin_chain(curr_angle[actuated])
            curr_pos = T_curr[-1][:3, 3]
            pos_error = endeffector_pos - curr_pos

            if np.linalg.norm(pos_error) < tol:
                break

            J_full = self._jacobian(T_curr)
            J_pos = J_full[:3, :]

            # Damped least squares: J^T (J J^T + lambda^2 I)^-1
            JJT = J_pos @ J_pos.T
            delta_angles = J_pos.T @ np.linalg.solve(
                JJT + damping**2 * np.eye(3), pos_error
            )

            # Clamp step size
            step_norm = np.linalg.norm(delta_angles)
            if step_norm > max_step:
                delta_angles *= max_step / step_norm

            # Clamp to [-pi, pi]
            angles = curr_angle[actuated] + delta_angles
            curr_angle[actuated] = np.arctan2(np.sin(angles), np.cos(angles))

        return curr_angle[actuated]

    def forward_kin(self, phi: np.ndarray, full=False) -> np.ndarray:
        if full == True:
            return np.array([t[:3, 3] for t in self._forward_kin_chain(phi)])
        else:
            return self._forward_kin(phi)[:3, 3]

    def inverse_kin(self, endeffector_pos: np.ndarray, curr_angle=None) -> np.ndarray:
        return self._inverse_kin(endeffector_pos, curr_angle)
