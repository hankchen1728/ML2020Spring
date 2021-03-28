import torch.nn as nn


def normal_train(
        model,
        optimizer,
        task,
        total_epochs,
        summary_epochs,
        device="cuda"):
    model.train()
    model.zero_grad()
    ceriation = nn.CrossEntropyLoss()
    losses = []
    loss = 0.0
    for epoch in range(summary_epochs):
        imgs, labels = next(task.train_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        ce_loss = ceriation(outputs, labels)

        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()

        loss += ce_loss.item()
        if (epoch + 1) % 50 == 0:
            loss = loss / 50
            print("\r", "train task {} [{}] loss: {:.3f}      ".format(
                task.name, (total_epochs + epoch + 1), loss), end=" ")
            losses.append(loss)
            loss = 0.0

    return model, optimizer, losses


def ewc_train(
        model,
        optimizer,
        task,
        total_epochs,
        summary_epochs,
        ewc,
        lambda_ewc,
        device="cuda"):
    model.train()
    model.zero_grad()
    ceriation = nn.CrossEntropyLoss()
    losses = []
    loss = 0.0
    for epoch in range(summary_epochs):
        imgs, labels = next(task.train_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        ce_loss = ceriation(outputs, labels)
        total_loss = ce_loss
        ewc_loss = ewc.penalty(model)
        total_loss += lambda_ewc * ewc_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss += total_loss.item()
        if (epoch + 1) % 50 == 0:
            loss = loss / 50
            print("\r", "train task {} [{}] loss: {:.3f}      ".format(
                task.name, (total_epochs + epoch + 1), loss), end=" ")
            losses.append(loss)
            loss = 0.0

    return model, optimizer, losses


def mas_train(
        model,
        optimizer,
        task,
        total_epochs,
        summary_epochs,
        mas_tasks,
        lambda_mas,
        alpha=0.8,
        device="cuda"):
    model.train()
    model.zero_grad()
    ceriation = nn.CrossEntropyLoss()
    losses = []
    loss = 0.0
    for epoch in range(summary_epochs):
        imgs, labels = next(task.train_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        ce_loss = ceriation(outputs, labels)
        total_loss = ce_loss
        mas_tasks.reverse()
        if len(mas_tasks) > 1:
            preprevious = 1 - alpha
            scalars = [alpha, preprevious]
            for mas, scalar in zip(mas_tasks[:2], scalars):
                mas_loss = mas.penalty(model)
                total_loss += lambda_mas * mas_loss * scalar
        elif len(mas_tasks) == 1:
            mas_loss = mas_tasks[0].penalty(model)
            total_loss += lambda_mas * mas_loss
        else:
            pass

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss += total_loss.item()
        if (epoch + 1) % 50 == 0:
            loss = loss / 50
            print("\r", "train task {} [{}] loss: {:.3f}      ".format(
                task.name, (total_epochs + epoch + 1), loss), end=" ")
            losses.append(loss)
            loss = 0.0

    return model, optimizer, losses


def scp_train(
        model,
        optimizer,
        task,
        total_epochs,
        summary_epochs,
        scp_tasks,
        lambda_scp,
        alpha=0.65,
        device="cuda"):

    model.train()
    model.zero_grad()
    ceriation = nn.CrossEntropyLoss()
    losses = []
    loss = 0.0
    for epoch in range(summary_epochs):
        imgs, labels = next(task.train_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        ce_loss = ceriation(outputs, labels)
        total_loss = ce_loss
        scp_tasks.reverse()
        if len(scp_tasks) > 1:
            preprevious = 1 - alpha
            scalars = [alpha, preprevious]
            for scp, scalar in zip(scp_tasks[:2], scalars):
                scp_loss = scp.penalty(model)
                total_loss += lambda_scp * scp_loss * scalar
        elif len(scp_tasks) == 1:
            scp_loss = scp_tasks[0].penalty(model)
            total_loss += lambda_scp * scp_loss
        else:
            pass

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss += total_loss.item()
        if (epoch + 1) % 50 == 0:
            loss = loss / 50
            print("\r", "train task {} [{}] loss: {:.3f}      ".format(
                task.name, (total_epochs + epoch + 1), loss), end=" ")
            losses.append(loss)
            loss = 0.0

    return model, optimizer, losses
