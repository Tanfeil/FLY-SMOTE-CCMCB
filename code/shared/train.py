from concurrent.futures import ProcessPoolExecutor

from code.shared import FlySmote
from code.shared.logger_config import configure_logger
from code.shared.structs import GANClientArgs, ClientArgs
from code.shared.train_client import train_gan_client, train_client


def train_gan_clients_and_average(gan_clients, global_gan_weights, config):
    num_global_samples = 0
    local_gan_weights_per_client = []
    num_samples_per_client = []

    with ProcessPoolExecutor(max_workers=config["workers"], initializer=configure_logger,
                             initargs=(config["verbose"],)) as executor:
        # Parallelisiertes Training für Clients
        client_args_list = [
            GANClientArgs(client_name, client_data, global_gan_weights, config["batch_size"],
                          config["classes"],
                          config["epochs_gan"], config["noise_dim"])
            for client_name, client_data in gan_clients.items()
        ]

        results = list(executor.map(train_gan_client, client_args_list))

    # accumulate results
    for client_name, gen_weights, disc_weights, num_samples in results:
        num_global_samples += num_samples
        num_samples_per_client.append(num_samples)
        local_gan_weights_per_client.append(gen_weights)
        gan_clients[client_name][1] = disc_weights

    scalar_per_client=[num_samples / num_global_samples for num_samples in
                                                                  num_samples_per_client]
    average_gan_weights = FlySmote.scale_and_sum_weights(local_gan_weights_per_client,
                                                         scalars=scalar_per_client)

    return average_gan_weights, gan_clients


def train_clients_and_average(clients, global_weights, early_stopping, lr_schedule, global_gan_weights,
                              config):
    # Calculate global data count for scaling
    # Calculate before so, the original size sets the impact for the global model.
    # So the synthetic created data does not higher the impact
    num_global_samples = sum([len(client) for client in clients.values()])
    local_weights_per_client = []
    num_samples_per_client = []

    # Parallel client training
    with ProcessPoolExecutor(max_workers=config["workers"], initializer=configure_logger,
                             initargs=(config["verbose"],)) as executor:
        args_list = [
            ClientArgs(
                client_name, client_data, global_weights, config["batch_size"], early_stopping,
                config["threshold"], config["k_value"], config["r_value"], config["epochs"],
                config["loss_function"], lr_schedule, config["metrics"], num_global_samples, config["g_value"],
                global_gan_weights, config["noise_dim"]
            )
            for client_name, client_data in clients.items()
        ]

        results = list(executor.map(train_client, args_list))

    for client_name, local_weights, num_samples in results:
        num_samples_per_client.append(num_samples)
        local_weights_per_client.append(local_weights)

    scalar_per_client = [num_samples / num_global_samples for num_samples in
                                                              num_samples_per_client]
    average_weights = FlySmote.scale_and_sum_weights(local_weights_per_client,
                                                     scalars=scalar_per_client)

    return average_weights
