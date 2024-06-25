from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger().setLevel(logging.ERROR)

class ServerStatus(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """

    BUSY = 1
    IDLE = 2


class UserStatus(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """

    BUSY = 1
    OFF = 2
    HOLD = 3


class EventType(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """

    ARRIVAL = 1
    RETRIAL = 2
    DONE = 3


class Event:
    """_summary_"""

    def __init__(self, event_type, accur_time=float("inf")) -> None:
        self.type = event_type
        self.accur_time = accur_time
        self.server_id = None
        self.user_id = None

    def set_server_id(self, server_id):
        """_summary_

        Args:
            server_id (_type_): _description_
        """
        self.server_id = server_id

    def set_user_id(self, user_id):
        """_summary_

        Args:
            user_id (_type_): _description_
        """
        self.user_id = user_id


class Server:
    """_summary_"""

    def __init__(self, server_id) -> None:
        self.id = server_id
        self.status = ServerStatus.IDLE
        self.remaining_service_time = 0
        self.next_available_time = 0
        self.user_id_served = None


class User:
    """_summary_"""

    def __init__(self, user_id) -> None:
        self.id = user_id
        self.status = UserStatus.OFF
        self.next_request_time = None
        self.remaining_service_time = None


class ServerStation:
    """_summary_"""

    def __init__(self, num_of_servers: int, service_rate: float) -> None:
        self.servers = {}
        for i in range(num_of_servers):
            self.servers[i] = Server(i)
        self.service_rate = service_rate
        self.idle_servers = set(range(num_of_servers))
        self.busy_servers = set()

    def find_next_service_completion_server(self, cur_time) -> Event:
        """_summary_

        Args:
            cur_time (_type_): _description_

        Returns:
            Event: _description_
        """
        next_event = Event(EventType.DONE)
        for server_id in self.busy_servers:
            if self.servers[server_id].next_available_time < next_event.accur_time:
                next_event.accur_time = self.servers[server_id].next_available_time
                next_event.set_server_id(server_id)
        if next_event.server_id is not None:
            next_event.set_user_id(self.servers[next_event.server_id].user_id_served)
        else:
            logging.warning(f"no event found from busy servers = {self.busy_servers}")
        return next_event

    def update(self, event, time_delta):
        """_summary_

        Args:
            event (_type_): _description_
            time_delta (_type_): _description_
        """
        # remove the server from busy servers and change the status
        self.servers[event.server_id].status = ServerStatus.IDLE
        self.servers[event.server_id].remaining_service_time = 0
        self.servers[event.server_id].next_available_time = event.accur_time
        self.busy_servers.remove(event.server_id)
        self.idle_servers.add(event.server_id)

        self.process_all_busy_servers(time_delta)

    def update_job(self, event, time_delta) -> float:
        """_summary_

        Args:
            event (_type_): _description_
            time_delta (_type_): _description_

        Returns:
            float: _description_
        """
        self.process_all_busy_servers(time_delta)
        server_id = self.idle_servers.pop()#random.sample(sorted(self.idle_servers), 1)
        # self.idle_servers.remove(server_id)
        self.servers[server_id].status = ServerStatus.BUSY
        self.busy_servers.add(server_id)

        self.servers[server_id].remaining_service_time = np.random.exponential(
            1 / self.service_rate
        )
        self.servers[server_id].next_available_time = (
            event.accur_time + self.servers[server_id].remaining_service_time
        )
        self.servers[server_id].user_id_served = event.user_id
        return server_id, self.servers[server_id].remaining_service_time

    def process_all_busy_servers(self, time_delta):
        """_summary_

        Args:
            time_delta (_type_): _description_

        Raises:
            Exception: _description_
        """
        for server_id in self.busy_servers:
            if self.servers[server_id].remaining_service_time <= time_delta:
                raise ValueError(
                    f"server = {server_id}'s remaining time={self.servers[server_id].remaining_service_time} <= time delta={time_delta}"
                )
            self.servers[server_id].remaining_service_time -= time_delta


class UserStation:
    """_summary_"""

    def __init__(
        self, num_of_users: int, arrival_rate, retrial_rate, service_rate
    ) -> None:
        self.num_of_users = num_of_users
        self.users = {}
        for i in range(num_of_users):
            self.users[i] = User(i)
        self.arrival_rate = arrival_rate
        self.retrial_rate = retrial_rate
        self.service_rate = service_rate
        self.initialize(num_of_users)

    def initialize(self, num_of_off_users, num_of_on_users=0, num_of_hold_users=0):
        """_summary_

        Args:
            num_of_on_users (_type_): _description_
            num_of_off_users (int, optional): _description_. Defaults to 0.
            num_of_hold_users (int, optional): _description_. Defaults to 0.
        """
        all_user_ids = list(range(self.num_of_users))
        random.shuffle(all_user_ids)
        self.on_users = set(all_user_ids[:num_of_on_users])
        self.off_users = set(
            all_user_ids[num_of_on_users : num_of_off_users + num_of_on_users]
        )
        self.hold_users = set(
            all_user_ids[
                num_of_off_users
                + num_of_on_users : num_of_off_users
                + num_of_on_users
                + num_of_hold_users
            ]
        )

        for user_id in self.off_users:
            self.users[user_id].status = UserStatus.OFF
            self.users[user_id].next_request_time = np.random.exponential(
                1 / self.arrival_rate
            )

        for user_id in self.on_users:
            self.users[user_id].status = UserStatus.BUSY
            self.users[user_id].remaining_service_time = np.random.exponential(
                1 / self.service_rate
            )

        for user_id in self.hold_users:
            self.users[user_id].status = UserStatus.HOLD
            self.users[user_id].next_request_time = np.random.exponential(
                1 / self.retrial_rate
            )

    def update_with_service(self, event, time_delta, service_time):
        """_summary_

        Args:
            event (_type_): _description_
            time_delta (_type_): _description_
            service_time (_type_): _description_

        Raises:
            ValueError: _description_
        """
        self.process_all_busy_users(time_delta, event)
        # remove user from current and reasign group
        if event.type == EventType.ARRIVAL:
            self.off_users.remove(event.user_id)
            self.on_users.add(event.user_id)
            self.users[event.user_id].status = UserStatus.BUSY
            self.users[event.user_id].remaining_service_time = service_time
            self.users[event.user_id].next_request_time = None
        elif event.type == EventType.RETRIAL:
            self.hold_users.remove(event.user_id)
            self.on_users.add(event.user_id)
            self.users[event.user_id].status = UserStatus.BUSY
            self.users[event.user_id].remaining_service_time = service_time
            self.users[event.user_id].next_request_time = None
        elif event.type == EventType.DONE:
            self.on_users.remove(event.user_id)
            self.off_users.add(event.user_id)
            self.users[event.user_id].status = UserStatus.OFF
            self.users[event.user_id].remaining_service_time = None
            self.users[event.user_id].next_request_time = (
                event.accur_time + np.random.exponential(1 / self.arrival_rate)
            )
        else:
            raise ValueError(f"wrong event type = {event.type}!")

    def update_without_service(self, event, time_delta):
        """_summary_

        Args:
            event (_type_): _description_
            time_delta (_type_): _description_

        Raises:
            ValueError: _description_
        """
        self.process_all_busy_users(time_delta, event)
        if event.type == EventType.ARRIVAL:
            self.off_users.remove(event.user_id)
            self.hold_users.add(event.user_id)
            self.users[event.user_id].status = UserStatus.HOLD
            self.users[event.user_id].remaining_service_time = None
            self.users[event.user_id].next_request_time = (
                event.accur_time + np.random.exponential(1 / self.retrial_rate)
            )
        elif event.type == EventType.RETRIAL:
            self.users[event.user_id].remaining_service_time = None
            self.users[event.user_id].next_request_time = (
                event.accur_time + np.random.exponential(1 / self.retrial_rate)
            )
        else:
            raise ValueError(f"wrong event type = {event.type}")

    def process_all_busy_users(self, time_delta, event):
        """_summary_

        Args:
            time_delta (_type_): _description_
        """
        if event.type == EventType.DONE:
            self.on_users.remove(event.user_id)
            self.off_users.add(event.user_id)
            self.users[event.user_id].next_request_time = event.accur_time + np.random.exponential(1/self.arrival_rate)
            self.users[event.user_id].remaining_service_time = None
        for user_id in self.on_users:
            self.users[user_id].remaining_service_time -= time_delta

    def find_next_user_event(self, cur_time):
        """_summary_

        Args:
            cur_time (_type_): _description_

        Returns:
            _type_: _description_
        """
        next_arrival = self.find_next_arrival_event(cur_time)
        next_retrial = self.find_next_retrial_event(cur_time)
        if next_arrival.accur_time < next_retrial.accur_time:
            return next_arrival
        else:
            return next_retrial

    def find_next_arrival_event(self, cur_time):
        """_summary_

        Args:
            cur_time (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        next_arrival_event = Event(EventType.ARRIVAL)
        for user_id in self.off_users:
            if (
                self.users[user_id].next_request_time is None
                or self.users[user_id].next_request_time < cur_time
            ):
                raise ValueError(
                    f"the user={user_id} in off state's remaining time is {self.users[user_id].next_request_time}"
                )
            if self.users[user_id].next_request_time < next_arrival_event.accur_time:
                next_arrival_event.accur_time = self.users[user_id].next_request_time
                next_arrival_event.set_user_id(user_id)
        if next_arrival_event.user_id is None:
            logging.warning(f"no arrival found from off users = {self.off_users}")
        return next_arrival_event

    def find_next_retrial_event(self, cur_time):
        """_summary_

        Args:
            cur_time (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        next_retrial_event = Event(EventType.RETRIAL)
        for user_id in self.hold_users:
            if (
                self.users[user_id].next_request_time is None
                or self.users[user_id].next_request_time < cur_time
            ):
                raise ValueError(
                    f"the user={user_id} in hold state's remaining time is {self.users[user_id].next_request_time}"
                )
            if self.users[user_id].next_request_time < next_retrial_event.accur_time:
                next_retrial_event.accur_time = self.users[user_id].next_request_time
                next_retrial_event.set_user_id(user_id)
        if next_retrial_event.user_id is None:
            logging.warning(f"no retrial event found from retrial users = {self.hold_users}")
        return next_retrial_event


class FiniteServiceSystem:
    """_summary_"""

    def __init__(
        self,
        num_of_servers: int,
        num_of_users: int,
        arrival_rate: float,
        service_rate: float,
        retrial_rate: float,
    ) -> None:
        self.server_station = ServerStation(num_of_servers, service_rate)
        self.user_station = UserStation(
            num_of_users, arrival_rate, retrial_rate, service_rate
        )

    def Run(self, total_time, start_count_time=0):
        """_summary_

        Args:
            total_time (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        cur_time = 0
        total_req_count = 0
        total_service_count = 0
        time_points = []
        result_avai = []
        while cur_time < total_time:
            next_event = self.find_next_event(cur_time)
            next_time_passed = next_event.accur_time - cur_time
            
            if next_event.type == EventType.DONE:
                self.user_station.process_all_busy_users(next_time_passed, next_event)
                self.server_station.update(next_event, next_time_passed)
                logging.info(f"time: {cur_time}, got event={next_event.type}, event_time={next_event.accur_time}, server={next_event.server_id} finish service for user={next_event.user_id}.")
            elif next_event.type in {EventType.ARRIVAL, EventType.RETRIAL}:
                total_req_count = total_req_count + 1 if cur_time > start_count_time else total_req_count
                if len(self.server_station.idle_servers) > 0:
                    total_service_count = total_service_count + 1 if cur_time > start_count_time else total_service_count
                    server_id, service_time = self.server_station.update_job(
                        next_event, next_time_passed
                    )
                    self.user_station.update_with_service(
                        next_event, next_time_passed, service_time
                    )
                    logging.info(f"time: {cur_time}, got event={next_event.type}, event_time={next_event.accur_time},assign user={next_event.user_id} to server={server_id} with service time={service_time}.")
                else:
                    self.user_station.update_without_service(
                        next_event, next_time_passed
                    )
                    logging.info(f"time: {cur_time}, got event={next_event.type}, event_time={next_event.accur_time},but user={next_event.user_id} cannot find available servers.")
            else:
                raise ValueError(f"wrong event type = {next_event.type}")
            
            cur_time = next_event.accur_time
            if cur_time > start_count_time:
                if total_req_count > 0:
                    availability = total_service_count / total_req_count
                    time_points.append(cur_time)
                    result_avai.append(availability)
        
        return availability, time_points, result_avai

    def find_next_event(self, cur_time):
        """_summary_

        Args:
            cur_time (_type_): _description_

        Returns:
            _type_: _description_
        """
        next_server_event = self.server_station.find_next_service_completion_server(
            cur_time
        )
        next_user_event = self.user_station.find_next_user_event(cur_time)
        if next_server_event.accur_time < next_user_event.accur_time:
            return next_server_event
        else:
            return next_user_event


# if __name__ == "__main__":
#     np.random.seed(316)
#     # 0.08665355416109355
#     m = 3
#     n = 5
#     lambda_rate = 1
#     mu = 1
#     nu = 1
#     finite_service_system = FiniteServiceSystem(m, n, lambda_rate, mu, nu)
#     from Metrics import compute_availability_handler
#     exact_alpha = compute_availability_handler(m, n, lambda_rate, mu, nu)
#     availability, time_points, result_avai = finite_service_system.Run(30000, 10000)
#     print(f"simulated availability = {availability} and exact availability={exact_alpha}")
#     plt.plot(time_points, result_avai, label="simulation", marker="*")
#     plt.axhline(y = exact_alpha, label="exact", color="red")
#     plt.xlabel("time")
#     plt.ylabel("availability")
#     plt.legend()
#     plt.show()
#     plt.close() 
