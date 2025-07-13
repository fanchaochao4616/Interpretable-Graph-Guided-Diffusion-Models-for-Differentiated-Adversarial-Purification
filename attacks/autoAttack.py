from autoattack import AutoAttack

def autoAttack(x, y, diffusion, network_clf, config,model = None):
    x = x.to(config.device)
    y = y.to(config.device)
    adversary = AutoAttack(network_clf, norm='Linf', eps=config.attack.ptb/255., version='standard', device=config.device)
    x_adv = adversary.run_standard_evaluation(x, y, bs=config.structure.bsize).to(config.device)
    return x_adv,None ,None