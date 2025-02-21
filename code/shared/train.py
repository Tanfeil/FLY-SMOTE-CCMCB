from concurrent.futures import ProcessPoolExecutor

from code.shared import FlySmote
from code.shared.logger_config import configure_logger
from code.shared.structs import GANClientArgs, ClientArgs
from code.shared.train_client import train_gan_client, train_client


def train_gan_clients_and_average(gan_clients, global_gan_weights, x_train, config):
    num_global_samples = 0
    scaled_local_gan_weights = []

    with ProcessPoolExecutor(max_workers=config["workers"], initializer=configure_logger, initargs=(config["verbose"],)) as executor:
        # Parallelisiertes Training für Clients
        client_args_list = [
            GANClientArgs(client_name, client_data, global_gan_weights, x_train, config["batch_size"], config["classes"],
                          config["epochs_gan"], config["noise_dim"])
            for client_name, client_data in gan_clients.items()
        ]

        results = list(executor.map(train_gan_client, client_args_list))

    # accumulate results
    for client_name, scaled_weights, disc_weights, num_local_samples in results:
        num_global_samples += num_local_samples
        scaled_local_gan_weights.append(scaled_weights)
        gan_clients[client_name][1] = disc_weights

    # aggregation of GAN-weights
    scaled = list(map(lambda x: FlySmote.scale_model_weights(x, 1 / num_global_samples),
                      scaled_local_gan_weights))

    average_gan_weights = [sum(weights) for weights in zip(*scaled)]
    # average_gan_weights = FlySmote.sum_scaled_weights(scaled)

    return average_gan_weights, gan_clients


def train_clients_and_average(clients, global_weights, x_train, early_stopping, lr_schedule, global_gan_weights,
                              config):
    # Calculate global data count for scaling
    # Calculate before so, the original size sets the impact for the global model.
    # So the synthetic created data does not higher the impact
    # TODO: does it make sense like this ?
    num_global_samples = sum([len(client) for client in clients.values()])

    # Parallel client training
    with ProcessPoolExecutor(max_workers=config["workers"], initializer=configure_logger,
                             initargs=(config["verbose"],)) as executor:
        args_list = [
            ClientArgs(
                client_name, client_data, global_weights, x_train, config["batch_size"], early_stopping,
                config["threshold"], config["k_value"], config["r_value"], config["epochs"],
                config["loss_function"], lr_schedule, config["metrics"], num_global_samples, config["g_value"],
                global_gan_weights, config["noise_dim"]
            )
            for client_name, client_data in clients.items()
        ]

        scaled_local_weights = list(executor.map(train_client, args_list))

    # Aggregate scaled weights and update global model
    average_weights = FlySmote.sum_scaled_weights(scaled_local_weights)

    return average_weights
