import torch

def lossfunction(predictions, rewards):
    returns = rewards.sum(dim=1)

    # Main task: predicting return at last timestep.
    # Essentiellement c'est le calcul de MSE
    main_loss = torch.mean(predictions[:, -1] - returns) ** 2

    # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
    # Prediction détient une dimension de plus alors il ajoute une dimensions avec returns[..., None]
    # Ça revient à faire returns[:, None] en une dimension
    aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
    # Combine losses
    # C'est nébuleux pour moi cette loss
    loss = main_loss + aux_loss * 0.5
    return loss