import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of training epochs")
    parser.add_argument('--num_clients', type=int, default=20,
                        help="number of clients")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='the fraction of clients: C')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help="the number of local epochs")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')
    parser.add_argument('--train_rule', type=str, default='FedAvg',
                        help='the training rule for personalized FL')
    parser.add_argument('--iid', action='store_true',
                        help='use iid dataset or not')
    parser.add_argument('--dataset', type=str, default='cifar',
                        help="name of dataset")
    parser.add_argument('--num_classes', type=int,
                        default=10, help="number of classes")
    parser.add_argument('--lam', type=float, default=1.0,
                        help='coefficient for reg term')
    parser.add_argument('--device', default='cpu',
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Hyper-parameter to avoid concentration')
    parser.add_argument('--noniid_percent', type=int, default=80,
                        help='Default set to 0.8 Set to 0.0 for IID.')
    args = parser.parse_args()
    return args
