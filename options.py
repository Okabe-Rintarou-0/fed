import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument("--num_clients", type=int, default=20, help="number of clients")
    parser.add_argument(
        "--frac", type=float, default=1.0, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_epoch", type=int, default=1, help="the number of local epochs"
    )
    parser.add_argument("--augment", action="store_true", help="data augment")
    parser.add_argument("--local_bs", type=int, default=50, help="local batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum")
    parser.add_argument(
        "--train_rule",
        type=str,
        default="FedAvg",
        help="the training rule for personalized FL",
    )
    parser.add_argument("--iid", action="store_true", help="use iid dataset or not")
    parser.add_argument("--dataset", type=str, default="cifar", help="name of dataset")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument(
        "--lam", type=float, default=1.0, help="coefficient for reg term"
    )
    parser.add_argument(
        "--lam1", type=float, default=1.0, help="coefficient for reg term"
    )
    parser.add_argument(
        "--lam2", type=float, default=1.0, help="coefficient for reg term"
    )
    parser.add_argument(
        "--lam3", type=float, default=1.0, help="coefficient for reg term"
    )
    parser.add_argument(
        "--lam4", type=float, default=1e-2, help="coefficient for reg term"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
    )
    parser.add_argument(
        "--record_client_data", action="store_true", help="Record label distribution"
    )
    parser.add_argument("--teacher_percent", default=0, type=float)
    parser.add_argument("--start_round", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="To use cuda, set to a specific GPU ID. Default set to use CPU.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="The parameter for the dirichlet distribution for data partitioning",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="Proximal term",
    )
    parser.add_argument(
        "--cont_weight",
        type=float,
        default=0.001,
        help="cont loss weight",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=64,
        help="gen batch size",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Hyper-parameter to avoid concentration",
    )
    parser.add_argument(
        "--noniid_percent",
        type=int,
        default=0.8,
        help="Default set to 0.8 Set to 0.0 for IID.",
    )
    parser.add_argument(
        "--model_het",
        type=bool,
        default=True,
        help="Use heterogeneous model",
    )
    parser.add_argument(
        "--domain_het",
        action="store_true",
        default=False,
        help="Use heterogeneous domain",
    )
    parser.add_argument(
        "--prob", action="store_true", default=False, help="Use probabilistic model"
    )
    parser.add_argument("--l2r_coeff", type=float, default=1e-2)
    parser.add_argument("--cmi_coeff", type=float, default=5e-4)
    parser.add_argument("--m1", type=int, default=3, help="Number of GMM components")
    parser.add_argument("--z_dim", type=int, default=128, help="Dimension of z")
    parser.add_argument(
        "--get_index", type=bool, default=False, help="Get index of loader"
    )
    parser.add_argument(
        "--em_iter", type=int, default=1, help="Iterations of EM algorithmn"
    )
    parser.add_argument("--iter_num", type=int, default=0)
    parser.add_argument("--base_dir", type=str, default="./", help="Base directory")
    args = parser.parse_args()

    if args.train_rule == "FedGMM":
        args.get_index = True

    if args.dataset == "pacs":
        args.domain_het = True
        args.iid = True
        args.num_classes = 7
    elif args.dataset == "rmnist":
        args.domain_het = True
        args.iid = True
        args.num_classes = 10
    elif args.dataset in ["mnist", "fmnist"]:
        args.num_classes = 10
    elif args.dataset == "emnist":
        args.num_classes = 62
    elif args.dataset in ["cifar10", "cifar"]:
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100

    args.teacher_clients = []
    return args
