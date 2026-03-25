# tests/test_real_model_train.py

from data.cifar10 import get_cifar_loaders
from train.trainer import train_finetune
from architectures.graph import ArchitectureGraph
from architectures.node import Node
from architectures.compiler import CompiledModel
from utils.logger import get_logger


logger = get_logger(__name__, logfile="logs/test_real_model.log")

def build_minimal_graph():
    graph = ArchitectureGraph()

    # Node 0: Conv
    conv_node = Node(
        node_id=0,
        op_type="conv",
        params={
            "in_channels": 3,
            "out_channels": 8,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1
        },
        parents=[]
    )

    # Node 1: ReLU
    relu_node = Node(
        node_id=1,
        op_type="relu",
        params={},
        parents=[0]
    )

    # Node 2: Flatten
    flatten_node = Node(
        node_id=2,
        op_type="flatten",
        params={},
        parents=[1]
    )

    # Node 3: Linear classifier
    linear_node = Node(
        node_id=3,
        op_type="linear",
        params={
            "in_features": 8 * 32 * 32,
            "out_features": 10
        },
        parents=[2]
    )

    graph.add_node(conv_node)
    graph.add_node(relu_node)
    graph.add_node(flatten_node)
    graph.add_node(linear_node)

    graph.set_output(3)

    return graph


def main():
    logger.info("Building minimal graph...")
    graph = build_minimal_graph()
    logger.info("Graph built successfully.")

    logger.info("Compiling model...")
    model = CompiledModel(graph)
    logger.info("Model compiled successfully.")

    logger.info("Loading CIFAR10 dataset...")
    train_loader, val_loader = get_cifar_loaders()
    logger.info("Dataset loaded.")

    logger.info("Starting training test...")
    error = train_finetune(
        model,
        train_loader,
        val_loader,
        device="cpu",
        epochs=1
    )

    logger.info(f"Final validation error: {error:.4f}")


if __name__ == "__main__":
    main()
