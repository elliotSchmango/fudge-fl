import flwr as fl
from flwr.common import parameters_to_ndarrays, FitIns

#shared mixin: captures global weights after every aggregation round
class _WeightCaptureMixin:
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.global_weights = parameters_to_ndarrays(aggregated_parameters)
        return aggregated_parameters, metrics


#standard federated averaging
class CapturingFedAvg(_WeightCaptureMixin, fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_weights = None


#byzantine-robust multi-krum aggregation
class CapturingKrum(_WeightCaptureMixin, fl.server.strategy.Krum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_weights = None


#federated optimization with proximal penalty
class CapturingFedProx(_WeightCaptureMixin, fl.server.strategy.FedProx):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_weights = None


#adaptive federated optimizer (Adam moments on server)
class CapturingFedAdam(_WeightCaptureMixin, fl.server.strategy.FedAdam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_weights = None


class FedDCStrategy(_WeightCaptureMixin, fl.server.strategy.FedAvg):
    """
    FedDC: Federated Learning with Local Drift Decoupling and Correction.
    Server-side aggregation is standard FedAvg weighted mean.
    Novelty lives in client-side local training via the injected feddc_alpha parameter:
      Client loss = CE(x) + <h_i, x - w> + (alpha/2)||x - w||^2
      Drift update: h_i += alpha * (x - w)   after each round
    Ref: FedDC (CVPR 2022).
    """
    def __init__(self, feddc_alpha: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feddc_alpha = feddc_alpha
        self.global_weights = None

    def configure_fit(self, server_round, parameters, client_manager):
        #pass feddc_alpha and round index to clients via config dict
        config = {
            "feddc_alpha": float(self.feddc_alpha),
            "server_round": int(server_round),
        }
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(client, fit_ins) for client in clients]


def get_strategy(name: str, num_clients: int) -> fl.server.strategy.Strategy:
    """factory: instantiate correct FL strategy by name with research-grade defaults"""
    name = name.lower().strip()
    #common flower arguments shared across all strategy constructors
    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
    )

    if name == "fedavg":
        return CapturingFedAvg(**common)

    elif name == "krum":
        f = 1 #assumed number of malicious clients
        m = num_clients - f - 2 #multi-krum selection count (standard formula)
        return CapturingKrum(**common, num_malicious_clients=f, num_clients_to_keep=m)

    elif name == "fedprox":
        #proximal_mu=0.1 is a standard starting point from the FedProx paper
        return CapturingFedProx(**common, proximal_mu=0.1)

    elif name == "fedadam":
        #eta: server learning rate, eta_l: client lr, tau: adaptivity param
        return CapturingFedAdam(
            **common, eta=1e-2, eta_l=1e-2, beta_1=0.9, beta_2=0.99, tau=1e-3
        )

    elif name == "feddc":
        #feddc_alpha controls drift correction strength; 0.1 from original paper
        return FedDCStrategy(**common, feddc_alpha=0.1)

    else:
        raise ValueError(
            f"unknown aggregator '{name}'. "
            "choose from: fedavg, krum, fedprox, fedadam, feddc"
        )
