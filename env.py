import math
import random
import numpy as np


class Env_MAPPO(object):
    area_side_length = 1500  # Side length of the square area
    total_duration = 500  # Total simulation duration
    time_step_fly = 2  # Time for UAV flight phase
    time_step_service = 6  # Time for UAV service phase
    time_step_total = time_step_fly + time_step_service  # Total time for one step
    max_steps = int(total_duration / time_step_total)  # Maximum steps per episode

    num_uavs = 3  # Number of UAVs
    uav_cpu_frequency = 1.5e9  # UAV CPU frequency (Hz)
    uav_initial_battery = 600000  # UAV initial battery capacity (Joules)
    uav_flight_speed = 20  # UAV flight speed (m/s)
    uav_chip_coeff = 10 ** (-27)  # UAV chip coefficient for energy consumption
    uav_base_power = 80  # UAV base power consumption (Watts)
    uav_fixed_height = 20.0  # UAV fixed flight height (meters)
    power_coeff_hover = 1.0  # Power coefficient for hovering
    power_coeff_level_flight = 0.8  # Power coefficient for level flight

    num_ugvs = 5  # Number of UGVs
    ugv_cpu_frequency = 1.5e9  # UGV CPU frequency (Hz)
    ugv_speed = 2.0  # UGV movement speed (m/s)
    ugv_uplink_power = 0.1  # UGV uplink transmission power (Watts)
    ugv_local_compute_ratio = 0.2  # Ratio of data computed locally by UGV
    ugv_min_height = 0.0  # UGV minimum height
    ugv_max_height = 5.0  # UGV maximum height

    ugv_data_lognormal_mu = 14.0  # Lognormal distribution mu for UGV data size
    ugv_data_lognormal_sigma = 1.5  # Lognormal distribution sigma for UGV data size
    min_data_clip_bits = 0.5 * 1048576  # Minimum data size clip (bits)
    max_data_clip_bits = 10e10  # Maximum data size clip (bits)
    cpu_cycles_per_bit = 1000  # CPU cycles required per bit of data

    bandwidth_mhz = 1  # Channel bandwidth in MHz
    channel_bandwidth = bandwidth_mhz * 10**6  # Channel bandwidth in Hz
    noise_power_los = 10 ** (-13)  # LoS noise power (Watts)
    noise_power_nlos = 10 ** (-11)  # NLoS noise power (Watts)
    ref_channel_gain = 1e-3  # Reference channel gain at 1m
    # --- LoS/NLoS Markov model parameters ---
    p_stay_los = 0.8  # Probability of staying in LoS state
    p_switch_los_to_nlos = 1.0 - p_stay_los  # Probability of switching from LoS to NLoS
    p_stay_nlos = 0.9  # Probability of staying in NLoS state
    p_switch_nlos_to_los = 1.0 - p_stay_nlos  # Probability of switching from NLoS to LoS

    # === Reinforcement Learning Interface ===
    action_bound = [-1, 1]  # Action bounds
    # Single agent action dimension: 4 (target UGV index, horizontal flight angle, horizontal flight distance ratio, offloading ratio)
    action_dimension = 4
    # Single agent observation dimension:
    # Own state: 1 (battery) + 3 (XYZ position) = 4
    # All UGV states: num_ugvs * (3 (XYZ position) + 1 (data amount) + 1 (channel)) = num_ugvs * 5
    # Total dimension = 4 + num_ugvs * 5
    state_dimension = 4 + num_ugvs * 5

    # === Reward Function Parameters ===
    delay_penalty_factor = 3.0  # Penalty factor for delay
    energy_penalty_factor = 0.001  # Penalty factor for energy consumption
    battery_depletion_penalty = 300  # Penalty for battery depletion

    # --- UGV initial position clustering parameters ---
    cluster_center_ratio = (0.9, 0.9)  # Relative position of cluster center (XY)
    cluster_radius_ratio = 0.05  # Relative size of cluster radius

    def __init__(self):
        self.uav_initial_positions = self._generate_initial_positions(self.num_uavs)  # Generate initial UAV positions
        self.reset()  # Reset the environment

    def _generate_initial_positions(self, num_entities, z_range=None):  # Generate initial positions for entities
        positions = []
        if num_entities > 0:
            cols = max(1, int(np.ceil(np.sqrt(num_entities))))  # Number of columns for grid placement
            rows = max(1, int(np.ceil(num_entities / cols)))  # Number of rows for grid placement
            xs = np.linspace(self.area_side_length * 0.1, self.area_side_length * 0.9, cols)  # X coordinates
            ys = np.linspace(self.area_side_length * 0.1, self.area_side_length * 0.9, rows)  # Y coordinates
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx < num_entities:
                        if z_range is not None:  # If z_range is specified (for UGVs)
                            z = np.random.uniform(z_range[0], z_range[1])
                            positions.append([xs[c], ys[r], z])
                        else:  # For UAVs (fixed height later)
                            positions.append([xs[c], ys[r]])
                        idx += 1
        while len(positions) < num_entities:  # If grid placement doesn't cover all entities
            if z_range is not None:
                pos = np.append(
                    np.random.uniform(0, self.area_side_length, size=2), np.random.uniform(z_range[0], z_range[1])
                )
            else:
                pos = np.random.uniform(0, self.area_side_length, size=2)
            positions.append(pos)
        return np.array(positions[:num_entities], dtype=np.float32)

    def _generate_lognormal_data_size(self):  # Generate UGV data size (bits)
        """Generates UGV data size (bits)"""
        data_size = np.random.lognormal(self.ugv_data_lognormal_mu, self.ugv_data_lognormal_sigma)
        return np.clip(data_size, self.min_data_clip_bits, self.max_data_clip_bits)

    def reset(self):  # Reset environment state
        # UAV state initialization, fixed height
        self.uav_positions = np.array(
            [[*pos[:2], self.uav_fixed_height] for pos in self._generate_initial_positions(self.num_uavs)],
            dtype=np.float32,
        )
        self.uav_battery_levels = np.full(self.num_uavs, float(self.uav_initial_battery), dtype=np.float32)

        # UGV state initialization, XY+Z
        cluster_center_x = self.area_side_length * self.cluster_center_ratio[0]
        cluster_center_y = self.area_side_length * self.cluster_center_ratio[1]
        cluster_radius = self.area_side_length * self.cluster_radius_ratio
        min_x = max(0, cluster_center_x - cluster_radius)
        max_x = min(self.area_side_length, cluster_center_x + cluster_radius)
        min_y = max(0, cluster_center_y - cluster_radius)
        max_y = min(self.area_side_length, cluster_center_y + cluster_radius)

        self.ugv_positions = np.zeros((self.num_ugvs, 3), dtype=np.float32)
        self.ugv_positions[:, 0] = np.random.uniform(min_x, max_x, size=self.num_ugvs)
        self.ugv_positions[:, 1] = np.random.uniform(min_y, max_y, size=self.num_ugvs)
        self.ugv_positions[:, 2] = np.random.uniform(self.ugv_min_height, self.ugv_max_height, size=self.num_ugvs)

        self.ugv_data_queue = np.array(
            [self._generate_lognormal_data_size() for _ in range(self.num_ugvs)], dtype=np.float32
        )
        self.ugv_channel_state = np.random.randint(0, 2, self.num_ugvs).astype(np.float32)  # 0 for LoS, 1 for NLoS

        self.current_step_number = 0
        return self._get_observation()

    def _get_observation(self):  # Get observations for all UAV agents
        observations = []
        norm_ugv_states_list = []
        max_data_norm = float(self.max_data_clip_bits)
        for i in range(self.num_ugvs):  # Normalize UGV states
            norm_ugv_pos_xyz = self.ugv_positions[i] / np.array(
                [self.area_side_length, self.area_side_length, self.ugv_max_height]
            )
            norm_data_size = np.clip(self.ugv_data_queue[i] / max_data_norm, 0.0, 1.0) if max_data_norm > 0 else 0.0
            channel_state = self.ugv_channel_state[i]
            norm_ugv_states_list.extend(
                [norm_ugv_pos_xyz[0], norm_ugv_pos_xyz[1], norm_ugv_pos_xyz[2], norm_data_size, channel_state]
            )
        norm_ugv_states = np.array(norm_ugv_states_list, dtype=np.float32)

        for uav_id in range(self.num_uavs):  # Construct observation for each UAV
            norm_battery = np.array([self.uav_battery_levels[uav_id] / self.uav_initial_battery], dtype=np.float32)
            norm_uav_pos_xyz = self.uav_positions[uav_id] / np.array(
                [self.area_side_length, self.area_side_length, self.uav_fixed_height]
            )
            own_state = np.concatenate((norm_battery, norm_uav_pos_xyz))  # UAV's own state
            state = np.concatenate((own_state, norm_ugv_states))  # Concatenate with all UGV states
            observations.append(state)
        return observations

    def _calculate_comm_compute_times_energy_multi(  # Calculate communication, computation times and energy
        self, uav_id, target_ugv_id, uav_offloading_action_ratio, current_uav_pos
    ):
        total_data_size = self.ugv_data_queue[target_ugv_id]
        if total_data_size <= 1e-6:  # No data to process
            return 0, 0, 0, 0, 0, 0
        data_local = total_data_size * self.ugv_local_compute_ratio  # Data processed locally by UGV
        time_local_compute = data_local / (self.ugv_cpu_frequency / self.cpu_cycles_per_bit) if data_local > 1e-6 else 0
        data_potential_for_uav = total_data_size * (1.0 - self.ugv_local_compute_ratio)  # Data that can be offloaded
        bits_to_offload = data_potential_for_uav * uav_offloading_action_ratio  # Actual bits to offload
        if bits_to_offload <= 1e-6:  # No data offloaded
            return 0, 0, time_local_compute, 0, 0, data_local
        ugv_pos = self.ugv_positions[target_ugv_id]
        channel_state = self.ugv_channel_state[target_ugv_id]  # 0 for LoS, 1 for NLoS
        dist = np.linalg.norm(current_uav_pos - ugv_pos)  # Distance between UAV and UGV
        dist = max(dist, 1e-3)  # Avoid division by zero
        noise_power = self.noise_power_nlos if channel_state == 1 else self.noise_power_los
        channel_gain = self.ref_channel_gain / (dist**2.5)  # Path loss model
        snr = max((self.ugv_uplink_power * channel_gain) / noise_power, 1e-9)  # Signal-to-Noise Ratio
        trans_rate = max(self.channel_bandwidth * math.log2(1 + snr), 1e-6)  # Transmission rate (Shannon-Hartley)
        time_transmission = bits_to_offload / trans_rate  # Transmission time
        time_edge_compute = bits_to_offload / (self.uav_cpu_frequency / self.cpu_cycles_per_bit)  # UAV computation time
        energy_compute_edge = (
            self.uav_chip_coeff * (self.uav_cpu_frequency**3) * time_edge_compute
        )  # UAV computation energy
        return (
            time_transmission,
            time_edge_compute,
            time_local_compute,
            energy_compute_edge,
            bits_to_offload,
            data_local,
        )

    def _update_ugv_states(self):  # Update UGV positions, data queues, and channel states
        for i in range(self.num_ugvs):
            move_angle = random.uniform(0, 2 * np.pi)  # Random movement angle
            move_dist = self.ugv_speed * self.time_step_total  # Movement distance
            dx = move_dist * math.cos(move_angle)
            dy = move_dist * math.sin(move_angle)
            dz = np.random.uniform(-0.5, 0.5)  # Small random fluctuation in Z
            self.ugv_positions[i, 0] += dx
            self.ugv_positions[i, 1] += dy
            self.ugv_positions[i, 2] += dz
            self.ugv_positions[i, 0:2] = np.clip(
                self.ugv_positions[i, 0:2], 0, self.area_side_length
            )  # Clip XY within area
            self.ugv_positions[i, 2] = np.clip(
                self.ugv_positions[i, 2], self.ugv_min_height, self.ugv_max_height
            )  # Clip Z
            if self.ugv_data_queue[i] <= 1e-6:  # If data queue is empty, generate new data
                self.ugv_data_queue[i] = self._generate_lognormal_data_size()
            current_channel_state = self.ugv_channel_state[i]  # Current channel state (0:LoS, 1:NLoS)
            rand_num = random.random()
            if current_channel_state == 0:  # If LoS
                if rand_num < self.p_switch_los_to_nlos:  # Switch to NLoS
                    self.ugv_channel_state[i] = 1.0
            else:  # If NLoS
                if rand_num < self.p_switch_nlos_to_los:  # Switch to LoS
                    self.ugv_channel_state[i] = 0.0

    def step(self, actions):  # Execute one environment step
        """
        Executes one environment time step.
        actions: list/array, shape=(num_uavs, action_dimension=4)
        Returns: observations (list), rewards (list), dones (list), infos (list)
        """
        self.current_step_number += 1
        global_terminal = False  # Flag for global termination
        total_step_delay = 0.0  # Total delay in this step
        total_step_energy = 0.0  # Total energy consumed in this step
        infos = [{} for _ in range(self.num_uavs)]  # Info dictionaries for each UAV
        ugv_processing_results = {}  # key: ugv_id, value: (data_local, data_offload)
        ugv_target_map = {}  # key: target_ugv_id, value: uav_id (tracks which UGV is targeted by which UAV)

        next_uav_positions = self.uav_positions.copy()  # Temporary storage for next UAV positions
        consumed_energy_this_step = np.zeros(self.num_uavs, dtype=np.float32)  # Energy consumed by each UAV

        for uav_id in range(self.num_uavs):  # Process actions for each UAV
            if self.uav_battery_levels[uav_id] <= 1e-3:  # If UAV battery is depleted
                infos[uav_id]["warning"] = "battery_depleted_cannot_act"
                continue
            action = np.clip(actions[uav_id], self.action_bound[0], self.action_bound[1])  # Clip actions
            action_normalized = (action + 1) / 2  # Normalize actions to [0, 1]
            target_ugv_id = min(int(action_normalized[0] * self.num_ugvs), self.num_ugvs - 1)  # Target UGV ID
            fly_angle = action_normalized[1] * 2 * np.pi  # Flight angle
            fly_dist_ratio = action_normalized[2]  # Flight distance ratio
            uav_offloading_action_ratio = action_normalized[3]  # Offloading ratio
            if self.ugv_data_queue[target_ugv_id] <= 1e-6:  # If target UGV has no data
                uav_offloading_action_ratio = 0.0
            # UAV fixed height, XY plane movement
            current_uav_pos = self.uav_positions[uav_id]
            intended_fly_distance = (
                fly_dist_ratio * self.uav_flight_speed * self.time_step_fly
            )  # Intended flight distance
            dx_uav = intended_fly_distance * math.cos(fly_angle)
            dy_uav = intended_fly_distance * math.sin(fly_angle)
            uav_pos_potential = current_uav_pos.copy()
            uav_pos_potential[0] += dx_uav
            uav_pos_potential[1] += dy_uav
            uav_pos_potential[2] = self.uav_fixed_height  # Maintain fixed height
            uav_pos_next_step = np.clip(  # Clip potential position within bounds
                uav_pos_potential,
                [0, 0, self.uav_fixed_height],
                [self.area_side_length, self.area_side_length, self.uav_fixed_height],
            )
            actual_fly_distance = np.linalg.norm(uav_pos_next_step[:2] - current_uav_pos[:2])  # Actual flight distance
            if actual_fly_distance > 1e-3:  # If UAV moved
                flight_power_coeff = self.power_coeff_level_flight
                energy_fly = self.uav_base_power * flight_power_coeff * self.time_step_fly
            else:  # If UAV hovered
                flight_power_coeff = self.power_coeff_hover
                energy_fly = self.uav_base_power * flight_power_coeff * self.time_step_fly
            if self.uav_battery_levels[uav_id] < energy_fly:  # If not enough battery to fly
                uav_pos_next_step = current_uav_pos  # Stay in current position
                energy_fly = 0
                flight_power_coeff = self.power_coeff_hover  # Assume hovering if cannot fly
                infos[uav_id]["warning"] = "low_battery_cannot_fly"

            # --- Simulate communication and computation ---
            can_service_this_ugv = (target_ugv_id not in ugv_target_map) and (self.ugv_data_queue[target_ugv_id] > 1e-6)
            if can_service_this_ugv:
                ugv_target_map[target_ugv_id] = uav_id  # Lock this UGV for this UAV

            energy_compute_edge = 0
            # Base energy consumption during service time always exists (using hover coefficient)
            energy_base_service = self.uav_base_power * self.power_coeff_hover * self.time_step_service
            delay_this_task = 0
            data_offloaded, data_local = 0, 0
            t_tr, t_edge, t_local = 0, 0, 0

            if can_service_this_ugv:
                t_tr, t_edge, t_local, energy_compute_edge, data_offloaded, data_local = (
                    self._calculate_comm_compute_times_energy_multi(
                        uav_id,
                        target_ugv_id,
                        uav_offloading_action_ratio,
                        uav_pos_next_step,  # Use next position for calculation
                    )
                )

                required_energy_comp_base = energy_compute_edge + energy_base_service
                available_after_flight = self.uav_battery_levels[uav_id] - energy_fly

                if (
                    available_after_flight < required_energy_comp_base
                ):  # If not enough battery for computation and base service
                    # Cancel edge computation
                    original_intended_offload = data_offloaded > 0
                    energy_compute_edge, data_offloaded, t_tr, t_edge = 0, 0, 0, 0
                    if original_intended_offload:
                        infos[uav_id]["warning"] = "low_battery_cannot_compute"

                    # Check if enough for base service energy
                    if available_after_flight < energy_base_service:
                        energy_base_service = max(0, available_after_flight)  # Consume remaining battery
                        infos[uav_id]["warning"] = "low_battery_reduce_base_service"
                    delay_this_task = t_local  # Only local computation delay
                else:
                    # Sufficient battery
                    if data_local > 1e-6 or data_offloaded > 1e-6:
                        delay_this_task = max(
                            t_local, t_tr + t_edge
                        )  # Task delay is max of local and (transmission + edge)
                    else:
                        delay_this_task = 0  # No data processed

                # Record processing results (only if successfully processed)
                if delay_this_task > 0 or data_local > 1e-6:
                    ugv_processing_results[target_ugv_id] = (data_local, data_offloaded)

            else:  # Cannot service this UGV (either targeted by another or no data)
                # Only incur base service energy consumption (check battery)
                available_after_flight = self.uav_battery_levels[uav_id] - energy_fly
                if available_after_flight < energy_base_service:
                    energy_base_service = max(0, available_after_flight)
                    infos[uav_id]["warning"] = "low_battery_reduce_base_service"

                delay_this_task = 0
                energy_compute_edge = 0
                # Add reason for not servicing
                if target_ugv_id in ugv_target_map:
                    infos[uav_id]["info"] = f"ugv_{target_ugv_id}_targeted"
                elif self.ugv_data_queue[target_ugv_id] <= 1e-6:
                    infos[uav_id]["info"] = f"ugv_{target_ugv_id}_no_data"

            # --- Calculate total energy and update UAV state ---
            # Total energy = flight energy + edge computation energy + base service energy
            total_energy_consumed_this_uav = energy_fly + energy_compute_edge + energy_base_service
            consumed_energy_this_step[uav_id] = total_energy_consumed_this_uav

            next_uav_positions[uav_id] = uav_pos_next_step  # Update temporary position

            total_step_delay += delay_this_task  # Accumulate total delay for this step

        # --- After processing all UAVs ---

        # Update all UAV batteries and positions
        self.uav_battery_levels -= consumed_energy_this_step
        self.uav_battery_levels = np.maximum(0, self.uav_battery_levels)  # Battery cannot be negative
        self.uav_positions = next_uav_positions

        total_step_energy = np.sum(consumed_energy_this_step)  # Total energy consumed in this step

        # Update data queues of serviced UGVs
        for ugv_id, (processed_local, processed_offload) in ugv_processing_results.items():
            self.ugv_data_queue[ugv_id] -= processed_local + processed_offload
            self.ugv_data_queue[ugv_id] = max(0, self.ugv_data_queue[ugv_id])  # Data queue cannot be negative

        # --- Calculate shared reward ---
        reward_component_delay = -self.delay_penalty_factor * total_step_delay
        reward_component_energy = -self.energy_penalty_factor * total_step_energy
        shared_reward = float(reward_component_delay + reward_component_energy)

        # --- Update all UGV states (movement, data generation, channel) ---
        self._update_ugv_states()

        # --- Check global termination conditions ---
        if np.any(self.uav_battery_levels <= 1e-3):  # If any UAV's battery is depleted
            global_terminal = True
            shared_reward -= self.battery_depletion_penalty  # Apply penalty
            term_reason = "battery_depleted_in_team"
        elif self.current_step_number >= self.max_steps:  # If max steps reached
            global_terminal = True
            term_reason = "max_steps_reached"
        else:
            term_reason = None

        if global_terminal:
            for i in range(self.num_uavs):
                infos[i]["termination_reason"] = term_reason

        # --- Prepare return values ---
        observations = self._get_observation()
        rewards = [shared_reward for _ in range(self.num_uavs)]  # All UAVs get the same shared reward
        dones = [global_terminal for _ in range(self.num_uavs)]  # All UAVs share the same done signal

        return observations, rewards, dones, infos
