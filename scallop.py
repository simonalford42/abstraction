import scallopy as scallop
import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModel(nn.Module):
    def __init__(self, n_colors):
        super().__init__()

        context = scallop.ScallopContext(provenance="difftopkproofs")

        # INPUTS (predicted by tau)
        context.add_relation("current_holding", (int,), input_mapping=[(i,) for i in range(n_colors)]) # what are we holding right now? unary predicate
        context.add_relation("current_domino", (int,int), # what dominos are in the world right now? binary predicate
                             input_mapping=[(i,j)
                                            for i in range(n_colors)
                                            for j in range(n_colors)])
        # this is also an input, which is the option that we are running
        context.add_relation("action", (int,), input_mapping=[(i,) for i in range(n_colors)])

        # OUTPUTS (predicted by world model, given inputs)
        context.add_relation("next_holding", (int,)) # what are we holding at the completion of the option
        context.add_relation("next_domino", (int,int)) # what dominos exit in the world at the completion of the option

        # WORLD MODEL
        # you will start holding n if you are currently holding m, a domino links m to n, and the option picks up n
        context.add_rule("next_holding(n) = current_holding(m) and current_domino(m,n) and action(n)")
        # there is a domino (n,m) if it previously existed and ~(action(m) and current_holding(n)). used demorgan's law to rewrite this
        context.add_rule("next_domino(n,m) = current_domino(n,m) and ~action(m)")
        context.add_rule("next_domino(n,m) = current_domino(n,m) and ~current_holding(n)")


        # think of these as neural networks which predict the true/false values of a predicate given the true/false values of the input predicates
        self.next_holding = context.forward_function("next_holding", output_mapping=[(i,) for i in range(n_colors)], retain_graph=True)
        self.next_domino = context.forward_function("next_domino", output_mapping=[(i,j) for i in range(n_colors) for j in range(n_colors)], retain_graph=True)

    def forward(self, action, holding, dominos):
        # given probabilities for the current action, the current held color, and the current dominos, return
        # probabilities for the new held color and the new dominos in existence
        h = self.next_holding(action=action.unsqueeze(0), current_holding=holding.unsqueeze(0), current_domino=dominos.unsqueeze(0))
        d = self.next_domino(action=action.unsqueeze(0), current_holding=holding.unsqueeze(0), current_domino=dominos.unsqueeze(0))

        return h.squeeze(0), d.squeeze(0)


def demo_world_model(demo_optimization=True):
    wm = WorldModel(10)

    dominos = torch.zeros(10,10)
    for n,m in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]:
        dominos[n,m] = 1

    holding = torch.zeros(10)
    holding[0] = 1

    for a in [1,3,4]:
        action = torch.zeros(10)
        action[a] = 1

        next_holding, next_dominos = wm(action, holding, dominos.view(-1))
        next_dominos = next_dominos.view(10,10)

        print("issuing action", a,
              "while holding",
              torch.argmax(holding).item(),
              "\n\tin the initial state\t",
              ", ".join(sorted([str((i,j)) for i in range(10) for j in range(10) if dominos[i,j] > 0.5])),
              "\n\tgives the new state\t",
              ", ".join(sorted([str((i,j)) for i in range(10) for j in range(10) if next_dominos[i,j] > 0.5])),
              "\n\tand the new holding\t",
              torch.argmax(next_holding).item())

        if not demo_optimization:
            continue

        # Can we infer the action?
        # Idea: optimize a distribution over one action was issued,
        #       subject to the constraint that we have to predict the correct dominos and correct next thing we are holding
        action_distribution = torch.zeros(10, requires_grad=True)

        optimizer = torch.optim.Adam([action_distribution])

        for step in range(1000):
            next_holding_prediction, next_dominos_prediction = wm(F.softmax(action_distribution),
                                                                  holding.detach(),
                                                                  dominos.detach().view(-1))
            next_dominos_prediction = next_dominos_prediction.view(10,10)

            loss = 0
            loss += (next_holding.detach()-next_holding_prediction).square().sum()
            loss += (next_dominos.detach()-next_dominos_prediction).square().sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print("\toptimization step", step,
                      "loss", loss.item(),
                      "predicted action", torch.argmax(action_distribution).item())

        # Can we infer what was being held?
        holding_distribution = torch.zeros(10, requires_grad=True)

        optimizer = torch.optim.Adam([holding_distribution])

        for step in range(1000):
            next_holding_prediction, next_dominos_prediction = wm(action,
                                                                  F.softmax(holding_distribution),
                                                                  dominos.view(-1))
            next_dominos_prediction = next_dominos_prediction.view(10,10)

            loss = 0
            loss += (next_holding.detach()-next_holding_prediction).square().sum()
            loss += (next_dominos.detach()-next_dominos_prediction).square().sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print("\toptimization step", step,
                      "loss", loss.item(),
                      "predicted holding", torch.argmax(holding_distribution).item())


        # Can we infer what dominos there were previously?
        domino_distribution = torch.zeros(10, 10, requires_grad=True)

        optimizer = torch.optim.Adam([domino_distribution])

        for step in range(1000):
            next_holding_prediction, next_dominos_prediction = wm(action,
                                                                  holding,
                                                                  torch.sigmoid(domino_distribution).view(-1))
            next_dominos_prediction = next_dominos_prediction.view(10,10)

            loss = 0
            loss += (next_holding.detach()-next_holding_prediction).square().sum()
            loss += (next_dominos.detach()-next_dominos_prediction).square().sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step%100 == 0:
                print("\toptimization step", step,
                      "loss", loss.item(),
                      "predicted dominos",
                      ", ".join(sorted([str((i,j)) for i in range(10) for j in range(10)
                                        if torch.sigmoid(domino_distribution)[i,j] > 0.5 ])))



        holding = next_holding
        dominos = next_dominos

demo_world_model(demo_optimization=True) # you can also set this to true for a more involved demo


def causal_consistency_loss(initial_state, final_state, option):
    """one possible way of doing this; this code doesn't actually run"""
    action_distribution = F.softmax(predict_abstract_action(initial_state, option))

    initial_state, final_state = tau(initial_state), tau(final_state)

    initial_holding, initial_dominos = initial_state
    final_holding, final_dominos = final_state

    initial_holding, final_holding = F.softmax(initial_holding), F.softmax(final_holding)
    initial_dominos, final_dominos = torch.sigmoid(initial_dominos), torch.sigmoid(final_dominos)

    predicted_holding, predicted_dominos = world_model(action_distribution,
                                                       initial_holding,
                                                       initial_dominos.view(-1))

    loss = ...
