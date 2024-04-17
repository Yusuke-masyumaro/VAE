import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
"in_channels" : params.in_channels,
"hidden_dim" : params.hidden_dim,
"residual_hidden_dim" : params.residual_hidden_dim,
"num_residual_layers" : params.num_residual_layers,
"embedding_dim" : params.embedding_dim,
"num_embeddings" : params.num_embeddings,
"lr = 2e-4" : params.lr = 2e-4,
"batch_size" : params.batch_size,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()