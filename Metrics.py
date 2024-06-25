from scipy.sparse import csc_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sympy as sp
from tqdm import tqdm
from FiniteServiceSystem import FiniteServiceSystem
import configurations as conf

ACCURACY_NUM = 50


def compute_state_space(num_of_servers: int, num_of_users: int) -> dict:
    """Computes the state space of the system and return a dictionary of state to index

    Args:
        num_of_servers (int):number of servers
        num_of_users (int): number of users

    Returns:
        states_to_index (dict): state to index
    """
    states_to_index = {}
    index = 0
    for x in range(num_of_servers + 1):
        for y in range(num_of_users + 1):
            if x + y > num_of_users:
                continue
            states_to_index[(x, y)] = index
            index += 1
    return states_to_index


def compute_paremater_matrix_of_balance_equation(
    num_of_servers: int,
    num_of_users: int,
    states_to_index: dict,
    arrival_rate: float,
    service_rate: float,
    retrial_rate: float,
):
    """Computes the parameter matrices of balance equations

    Args:
        num_of_servers (int): number of servers
        num_of_users (int):
        states_to_index (dict): state to index
        arrival_rate (float):
        service_rate (float):
        retrial_rate (float):

    Returns:
        a(sparse matrix), b(numpy array):
    """
    cols = []
    rows = []
    data = []

    for x in range(num_of_servers + 1):
        for y in range(num_of_users + 1):
            if x + y > num_of_users:
                continue
            cur_row_index = states_to_index[(x, y)]
            rows.append(cur_row_index)
            cols.append(cur_row_index)
            data.append(
                -(
                    x * service_rate
                    + y * retrial_rate
                    + (num_of_users - x - y) * arrival_rate
                )
            )
            if states_to_index.get((x + 1, y)):
                rows.append(cur_row_index)
                cols.append(states_to_index[(x + 1, y)])
                data.append((x + 1) * service_rate)
            if states_to_index.get((x - 1, y)):
                rows.append(cur_row_index)
                cols.append(states_to_index[(x - 1, y)])
                data.append((num_of_users - x - y + 1) * arrival_rate)
            if states_to_index.get((x + 1, y + 1)):
                rows.append(cur_row_index)
                cols.append(states_to_index[(x + 1, y + 1)])
                data.append((y + 1) * retrial_rate)
    dim_of_matrix = len(states_to_index)
    # add the all one vector to the last row
    rows.extend([dim_of_matrix] * dim_of_matrix)
    cols.extend(list(range(dim_of_matrix)))
    data.extend([1] * dim_of_matrix)

    param_mtx = csc_matrix(
        (data, (rows, cols)), shape=(dim_of_matrix + 1, dim_of_matrix)
    )
    b = np.zeros((dim_of_matrix + 1, 1))
    b[-1, 0] = 1

    # param_mtx = csc_matrix((data, (rows, cols)), shape = (dim_of_matrix, dim_of_matrix))
    # b = np.zeros((dim_of_matrix, 1))
    return param_mtx, b


def compute_stationary_distribution(a, b):
    """Given the linear system a * x = b, this function is to solve x
    Args:
        a (sparse matrix): the parameter matrix of balance equation
        b (numpy array): the right hand side vector of linear system ax = b used to solve the stationary distribution x

    Returns:
        numpy array:stationary distribution
    """
    # ata = csc_matrix.dot(a.transpose(), a)
    # atb = csc_matrix.dot(a.transpose(), b)
    # return linalg.solve(ata.toarray(), atb)
    # Pi = linalg.solve(a.toarray(), b)
    # return Pi/Pi.sum()
    # Xinv = np.linalg.pinv(param_matrix.toarray()) @ b
    # Xlst = np.linalg.lstsq(param_matrix.toarray(), b)[0]
    # # from sklearn.linear_model import LinearRegression
    # # lr = LinearRegression()
    # # lr.fit(param_matrix.toarray(), b)
    # # Xlr = np.array(lr.coef_).reshape((-1, 1))
    # print(np.abs(np.matmul(param_matrix.toarray(), Xsolv) - b).sum())
    # print(np.abs(np.matmul(param_matrix.toarray(), Xinv) - b).sum())
    # print(np.abs(np.matmul(param_matrix.toarray(), Xlst) - b).sum())

    a_mat_s = sp.Matrix(a.toarray())
    a_mat_s = a_mat_s.applyfunc(lambda x: sp.Float(x, ACCURACY_NUM))
    a_mat_transpose = a_mat_s.transpose()
    ata_inv = (a_mat_transpose * a_mat_s).inv()

    b_vec = sp.Matrix([sp.Float(i, ACCURACY_NUM) for i in b])
    atb = a_mat_transpose * b_vec
    return ata_inv * atb


def compute_expected_num_of_busy_users(
    num_of_servers, num_of_users, stationary_distribution, states_to_index
):
    """Computes expected number of busy servers E[X]

    Args:
        num_of_servers (int):
        num_of_users (int):
        stationary_distribution (vector of float):
        states_to_index (dict):

    Returns:
        expected_users (float):
    """
    expected_num_busy_users = sp.Float(0, ACCURACY_NUM)
    for x in range(num_of_servers + 1):
        for y in range(num_of_users + 1):
            if x + y > num_of_users:
                continue
            expected_num_busy_users += (
                x * stationary_distribution[states_to_index[(x, y)]]
            )
    return expected_num_busy_users


def compute_availability(
    expected_busy_users, arrival_rate, service_rate, retrial_rate, num_of_users
):
    """_summary_

    Args:
        expected_busy_users (_type_): _description_
        arrival_rate (_type_): _description_
        service_rate (_type_): _description_
        retrial_rate (_type_): _description_
        num_of_users (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (expected_busy_users * service_rate) / (
        expected_busy_users * service_rate
        + (
            num_of_users
            - (arrival_rate + service_rate) / arrival_rate * expected_busy_users
        )
        * retrial_rate
    )


def compute_availability_handler(
    num_of_servers, num_of_users, arrival_rate, service_rate, retrial_rate, simulation
):
    """_summary_

    Args:
        num_of_servers (_type_): _description_
        num_of_users (_type_): _description_
        arrival_rate (_type_): _description_
        service_rate (_type_): _description_
        retrial_rate (_type_): _description_

    Returns:
        _type_: _description_
    """
    if arrival_rate != retrial_rate:
        if not simulation:
            states_to_index = compute_state_space(num_of_servers, num_of_users)
            param_matrix, b = compute_paremater_matrix_of_balance_equation(
                num_of_servers,
                num_of_users,
                states_to_index,
                arrival_rate,
                service_rate,
                retrial_rate,
            )
            stationary_distribution = compute_stationary_distribution(param_matrix, b)
            expected_busy_users = compute_expected_num_of_busy_users(
                num_of_servers, num_of_users, stationary_distribution, states_to_index
            )
            availability = compute_availability(
                expected_busy_users,
                arrival_rate,
                service_rate,
                retrial_rate,
                num_of_users,
            )
        else:
            finite_service_system = FiniteServiceSystem(
                num_of_servers, num_of_users, arrival_rate, service_rate, retrial_rate
            )
            availability, _, _ = finite_service_system.Run(
                conf.TOTAL_RUN_TIME, conf.START_COUNT_TIME
            )
    else:
        availability = 0
        for m in range(num_of_servers + 1):
            availability = m / (
                m
                + (1 - availability)
                * (num_of_users - m)
                * (arrival_rate / service_rate)
            )
    return availability


def find_minimum_num_of_servers_for_given_avilability(
    service_level: float,
    num_of_users,
    arrival_rate,
    service_rate,
    retrial_rate,
    lower_bound_of_servers=1,
    upper_bound_of_servers=None,
):
    """_summary_

    Args:
        service_level (float): _description_
        num_of_users (_type_): _description_
        arrival_rate (_type_): _description_
        service_rate (_type_): _description_
        retrial_rate (_type_): _description_
        lower_bound_of_servers (int, optional): _description_. Defaults to 1.
        upper_bound_of_servers (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if upper_bound_of_servers is None:
        upper_bound_of_servers = num_of_users
    if upper_bound_of_servers < lower_bound_of_servers:
        raise ValueError(
            f"Error: the given upper_bound_of_servers: {upper_bound_of_servers} is smaller than the given lower_bound_of_servers: {lower_bound_of_servers}"
        )
    while lower_bound_of_servers < upper_bound_of_servers:
        mid = (
            lower_bound_of_servers
            + (upper_bound_of_servers - lower_bound_of_servers) // 2
        )
        availability = compute_availability_handler(
            mid,
            num_of_users,
            arrival_rate,
            service_rate,
            retrial_rate,
            conf.USE_SIMULATION,
        )
        if availability < service_level:
            lower_bound_of_servers = mid + 1
        else:
            upper_bound_of_servers = mid
    return lower_bound_of_servers


def compute_general_upper_bound_with_given_availability(
    num_of_users, service_rate, retrial_rate, availability
):
    """_summary_

    Args:
        num_of_users (_type_): _description_
        service_rate (_type_): _description_
        retrial_rate (_type_): _description_
        availability (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.ceil(
        availability
        * retrial_rate
        * num_of_users
        / (availability * retrial_rate + (1 - availability) * service_rate)
    )


def compute_general_lower_bound_with_given_availability(
    num_of_users, arrival_rate, service_rate, retrial_rate, availability
):
    """_summary_

    Args:
        num_of_users (_type_): _description_
        arrival_rate (_type_): _description_
        service_rate (_type_): _description_
        retrial_rate (_type_): _description_
        availability (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.ceil(
        arrival_rate
        / (
            arrival_rate
            + service_rate
            + (1 - availability)
            / availability
            * (arrival_rate * service_rate / retrial_rate)
        )
        * num_of_users
    )


def compute_upper_bound_for_large_system_with_availability(
    num_of_users, arrival_rate, service_rate, retrial_rate, availability
):
    """_summary_

    Args:
        num_of_users (_type_): _description_
        arrival_rate (_type_): _description_
        service_rate (_type_): _description_
        retrial_rate (_type_): _description_
        availability (_type_): _description_

    Returns:
        _type_: _description_
    """
    c_underline = arrival_rate / (
        arrival_rate
        + service_rate
        + (1 - availability)
        / availability
        * (arrival_rate * service_rate / retrial_rate)
    )
    return np.ceil(
        c_underline * num_of_users
        + 2
        * service_rate
        * np.sqrt(c_underline / (arrival_rate + service_rate))
        / (np.sqrt(arrival_rate) - np.sqrt(c_underline * (arrival_rate + service_rate)))
    )


def compute_lower_bound_for_large_system_with_availability(
    num_of_users, arrival_rate, service_rate, retrial_rate, availability
):
    """_summary_

    Args:
        num_of_users (_type_): _description_
        arrival_rate (_type_): _description_
        service_rate (_type_): _description_
        retrial_rate (_type_): _description_
        availability (_type_): _description_

    Returns:
        _type_: _description_
    """
    c_underline = arrival_rate / (
        arrival_rate
        + service_rate
        + (1 - availability)
        / availability
        * (arrival_rate * service_rate / retrial_rate)
    )
    return np.ceil(
        c_underline * num_of_users
        + service_rate
        * c_underline
        / (
            service_rate * c_underline
            + retrial_rate * (1 - c_underline + 1 / num_of_users)
        )
    )


def compute_threshold_of_large_system(
    arrival_rate, service_rate, retrial_rate, availability
):
    """_summary_

    Args:
        arrival_rate (_type_): _description_
        service_rate (_type_): _description_
        retrial_rate (_type_): _description_
        availability (_type_): _description_

    Returns:
        _type_: _description_
    """
    c_underline = arrival_rate / (
        arrival_rate
        + service_rate
        + (1 - availability)
        / availability
        * (arrival_rate * service_rate / retrial_rate)
    )
    return np.ceil(
        2
        * service_rate
        / (np.sqrt(arrival_rate) - np.sqrt(c_underline * (arrival_rate + service_rate)))
        ** 2
    )


def compute_upper_bound_with_availability_handler(
    num_of_users,
    arrival_rate,
    service_rate,
    retrial_rate,
    availability,
    threshold_of_large_system,
):
    """_summary_

    Args:
        num_of_users (_type_): _description_
        arrival_rate (_type_): _description_
        service_rate (_type_): _description_
        retrial_rate (_type_): _description_
        availability (_type_): _description_
        threshold_of_large_system (_type_): _description_

    Returns:
        _type_: _description_
    """
    if num_of_users >= threshold_of_large_system:
        return compute_upper_bound_for_large_system_with_availability(
            num_of_users, arrival_rate, service_rate, retrial_rate, availability
        )
    else:
        return compute_general_upper_bound_with_given_availability(
            num_of_users, service_rate, retrial_rate, availability
        )


def ComputeAllMinServers(params, all_n):
    results_min_servers = np.zeros(len(all_n)) + np.nan
    results_min_servers_lb = np.zeros(len(all_n)) + np.nan
    results_min_servers_ub = np.zeros(len(all_n)) + np.nan
    threshold = compute_threshold_of_large_system(
        params["arrival_rate"],
        params["service_rate"],
        params["retrial_rate"],
        params["given_alpha"],
    )

    for i, n in tqdm(enumerate(all_n)):
        results_min_servers[i] = find_minimum_num_of_servers_for_given_avilability(
            params["given_alpha"],
            n,
            params["arrival_rate"],
            params["service_rate"],
            params["retrial_rate"],
        )
        results_min_servers_lb[i] = (
            compute_lower_bound_for_large_system_with_availability(
                n,
                params["arrival_rate"],
                params["service_rate"],
                params["retrial_rate"],
                params["given_alpha"],
            )
        )
        results_min_servers_ub[i] = compute_upper_bound_with_availability_handler(
            n,
            params["arrival_rate"],
            params["service_rate"],
            params["retrial_rate"],
            params["given_alpha"],
            threshold,
        )
    return (
        results_min_servers,
        results_min_servers_lb,
        results_min_servers_ub,
        threshold,
    )


def PlotAllPerformances(
    params,
    all_n,
    results_min_servers,
    results_min_servers_lb,
    results_min_servers_ub,
    threshold,
    x_step_size,
    y_step_size,
):
    plt.plot(all_n, results_min_servers, label="Exact")
    plt.plot(all_n, results_min_servers_lb, label="lower_bound")
    plt.plot(all_n, results_min_servers_ub, label="upper_bound")
    plt.title(
        rf"$\lambda$={params['arrival_rate']}, $\mu$={params['service_rate']}, $\nu$={params['retrial_raer']}, threshold={int(threshold)}, $\bar\alpha$={params['given_alpha']}"
    )
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(x_step_size))
    plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(y_step_size))
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    params = {}
    params["arrival_rate"] = 1
    params["service_rate"] = 1
    params["retrial_rate"] = 2
    params["given_alpha"] = 0.9

    all_n = np.arange(2, 3000, 200)
    results_min_servers, results_min_servers_lb, results_min_servers_ub, threshold = (
        ComputeAllMinServers(params, all_n)
    )
    PlotAllPerformances(
        params,
        all_n,
        results_min_servers,
        results_min_servers_lb,
        results_min_servers_ub,
        threshold,
        20,
        2,
    )
