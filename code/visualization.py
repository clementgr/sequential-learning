import numpy as np
import matplotlib.pyplot as plt


def plot_p(p, save_fig=False, save_name='p1_3b', player='p', algo=''):
    T = len(p)
    x_val = range(1, T+1)
    rock_val = [pt[0] for pt in p]
    paper_val = [pt[1] for pt in p]
    cisors_val = [pt[2] for pt in p]
    f, ax = plt.subplots(1,3, figsize=(15,5), sharey='row')
    ax[0].plot(x_val, rock_val, color='grey')
    ax[0].set_title('rock')
    ax[1].plot(x_val, paper_val, color='skyblue')
    ax[1].set_title('paper')
    ax[2].plot(x_val, cisors_val, color='salmon')
    ax[2].set_title('scissors')
    for i in range(3):
        ax[i].set_xlabel('time $t$')
    ax[0].set_ylabel(f'${player}_t$')
    f.suptitle(f'Evolution of ${player}_t$ as a function of $t$ ({algo})', y=1.05)
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_loss(loss, save_fig=False, save_name='p1_3a', algo=''):
    plt.figure(figsize=(10,5))
    T = len(loss)
    plt.plot(range(1,T+1), loss, color='#9D2CB8')
    plt.xlabel('time $t$')
    plt.ylabel('loss')
    plt.title(f'Evolution of the loss with time ({algo})')
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_avg_loss(loss, save_fig=False, save_name='p1_3c', algo=''):
    plt.figure(figsize=(10,5))
    T = len(loss)
    plt.plot(range(1,T+1), np.cumsum(loss) / np.arange(1,T+1), color='#9D2CB8')
    plt.plot(range(1,T+1), loss, color='black', alpha=0.1)
    plt.xlabel('time $t$')
    plt.ylabel('average loss')
    plt.title(f'Evolution of the average loss with time ({algo})')
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_cumul_regret(cum_regret, save_fig=False, save_name='p1_3d', algo=''):
    fig = plt.figure(figsize=(10,5))
    plt.plot(range(1,len(cum_regret)+1), cum_regret, color='#2CB0B8')
    plt.xlabel('time $t$')
    plt.ylabel('cumulative regret')
    plt.title(f'Cumulative regret evolution over time ({algo})')
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_avg_loss_minmax(losses, save_fig=False, save_name='p1_3e', algo=''):
    n = len(losses)
    losses = np.array(losses)
    T = losses.shape[1]
    cumsum_loss = np.cumsum(losses, axis=1) / np.arange(1,T+1)
    fig = plt.figure(figsize=(10,5))
    plt.plot(range(T), np.mean(cumsum_loss, axis=0) , color='#70A2A5', label='avg')
    plt.plot(range(T), np.amax(cumsum_loss, axis=0) , color='#C0335C', label='max')
    plt.plot(range(T), np.amin(cumsum_loss, axis=0) , color='#61AD70', label='min')
    for i in range(n):
        plt.plot(range(T), cumsum_loss[i], alpha=0.05, color='black')
    plt.legend()
    plt.xlabel('time $t$')
    plt.ylabel('loss')
    plt.title(f'Loss over {n} simulations ({algo})', y=1.05)
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_avg_p(p, save_fig=False, save_name='p1_4b_avg', player='p', algo=''):
    n = len(p)
    p = np.array(p)
    T = p.shape[1]
    f, ax = plt.subplots(1,2, figsize=(15,5))
    cum_time = np.arange(1,T+1)
    cum_time_3d = np.repeat(cum_time[:, np.newaxis], 3, axis=1)
    cumsum_p = np.cumsum(p, axis=1) / cum_time_3d
    ax[0].plot(range(1,T+1), np.mean(cumsum_p[...,0], axis=0), color='grey', label='rock')
    ax[0].plot(range(1,T+1), np.mean(cumsum_p[...,1], axis=0), color='skyblue', label='paper')
    ax[0].plot(range(1,T+1), np.mean(cumsum_p[...,2], axis=0), color='salmon', label='scissors')
    ax[0].legend()
    ax[0].set_xlabel('time $t$')
    ax[0].set_ylabel(f'average {player}')
    ax[0].set_title(f'Average probabilities over time')
    ax[0].set_xscale('log')
    for i in range(n):
        ax[0].plot(range(1,T+1), cumsum_p[i,:,0], alpha=0.1, color='grey')
        ax[0].plot(range(1,T+1), cumsum_p[i,:,1], alpha=0.1, color='skyblue')
        ax[0].plot(range(1,T+1), cumsum_p[i,:,2], alpha=0.1, color='salmon')
        
    tot_prob = np.mean(cumsum_p[...,0], axis=0) + np.mean(cumsum_p[...,1], axis=0) + np.mean(cumsum_p[...,2], axis=0)
    tot_prob = [np.round(p,4) for p in tot_prob]
    ax[1].plot(range(1,T+1), tot_prob, color='black')
    ax[1].set_xlabel('time $t$')
    ax[1].set_ylabel('probabilities sum')
    ax[1].set_title('Probabilities sum over time')

    f.suptitle(f'Average computed over {n} simulations ({algo})', y=1.05)
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_avg_p_in_individual_boxes(p, save_fig=False, save_name='p1_4b_indiv', player='p', algo=''):
    n = len(p)
    p = np.array(p)
    T = p.shape[1]
    titles = ['rock', 'paper', 'scissors']
    c = ['grey', 'skyblue', 'salmon']
    f, ax = plt.subplots(1,3, figsize=(15,5))
    cum_time = np.arange(1,T+1)
    cum_time_3d = np.repeat(cum_time[:, np.newaxis], 3, axis=1)
    cumsum_p = np.cumsum(p, axis=1) / cum_time_3d
    for k in range(3):
        ax[k].plot(range(1,T+1), np.mean(cumsum_p[...,k], axis=0) , color=c[k])
        ax[k].set_xlabel('time $t$')
        ax[k].set_ylabel(f'average {player}')
        ax[k].set_title(f'{titles[k]}')
        ax[k].set_xscale('log')
        for i in range(n):
            ax[k].plot(range(1,T+1), cumsum_p[i,:,k], alpha=0.05, color='black')
    f.suptitle(f'Average computed over {n} simulations ({algo})', y=1.05)
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_p_log_scale(p, save_fig=False, save_name='p1_4b_logscale_p', player='p', algo=''):
    p_init = 1/3 * np.ones(3)
    n = len(p)
    p = np.array(p)
    T = p.shape[1]
    cum_time = np.arange(1,T+1)
    cum_time_3d = np.repeat(cum_time[:, np.newaxis], 3, axis=1)
    cumsum_p = np.cumsum(p, axis=1) / cum_time_3d
    euclidian_norm_quantity = np.linalg.norm(cumsum_p - p_init, axis=2)
    plt.figure(figsize=(10,5))
    plt.plot(range(1,T+1), np.mean(euclidian_norm_quantity, axis=0), color='#6177AD')
    for i in range(n):
        plt.plot(range(1,T+1), euclidian_norm_quantity[i], alpha=0.05, color='black')
    plt.xlabel(f'time $t$')
    if player == 'p':
        plt.ylabel(r'$\vert \vert \bar{p}_t - (1/3, 1/3, 1/3) \vert \vert_2$')
        plt.title(r'$\vert \vert \bar{p}_t - (1/3, 1/3, 1/3) \vert \vert_2$ over time')
    else:
        plt.ylabel(r'$\vert \vert \bar{q}_t - (1/3, 1/3, 1/3) \vert \vert_2$')
        plt.title(r'$\vert \vert \bar{q}_t - (1/3, 1/3, 1/3) \vert \vert_2$ over time')
    plt.xscale('log')
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_regret_hist(regrets, save_fig=False, save_name='p2_1c', algo=''):
    n_run = len(regrets)
    fig = plt.figure(figsize=(10,5))
    plt.hist(regrets, density=False, color='#2CB0B8')
    plt.ylabel('counts')
    plt.xlabel('regret')
    plt.title(f'Histogram of $R_T$ of {algo} over {n_run} runs')
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_mean_regret(cum_regrets, alpha=0.05, save_fig=False, save_name='p2_1d', title='', algo=''):
    fig = plt.figure(figsize=(10,5))
    n, T = cum_regrets.shape
    plt.plot(range(1,T+1), np.mean(cum_regrets, axis=0), color='#2CB0B8', linewidth=2, label='avg')
    for i in range(n):
        plt.plot(range(1,T+1), cum_regrets[i], color='black', alpha=alpha, linewidth=0.5)
    plt.xlabel('time $t$')
    plt.ylabel('cumulative regret')
    plt.title(f'{title} ({algo})')
    plt.legend()
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_multiple_avg_regrets(regret_dict, save_fig=False, save_name='p2_2g_ftl_vs_ucb', title=''):
    fig = plt.figure(figsize=(10,5))
    n, T = regret_dict[list(regret_dict.keys())[0]][0].shape
    for name, (cum_regret, color) in regret_dict.items():
        plt.plot(range(1,T+1), np.mean(cum_regret, axis=0), color=color, linewidth=2, label=f'{name}')
    plt.xlabel('time $t$')
    plt.ylabel('cumulative regret')
    plt.title(f'{title}')
    plt.legend()
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_final_regret_sigma_square(regrets, sigma_squares, save_fig=False, save_name='p2_2h', title=''):
    fig = plt.figure(figsize=(10,5))
    plt.plot(sigma_squares, np.mean(regrets, axis=0).tolist(), color='#9536E5', linewidth=2, linestyle='None', marker='x')
    plt.xticks(sigma_squares, labels=[f'$\sigma_{i+1}^2$' for i, sigma in enumerate(sigma_squares)])
    plt.xlabel('$\sigma^2$')
    plt.ylabel('final regret')
    plt.title(f'{title}')
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()


def plot_final_regret_eta(etas, final_regrets, save_fig=False, save_name='p1_3f', title='', color='black'):
    plt.figure(figsize=(10,5))
    plt.plot(etas, final_regrets, color=color, linestyle='None', marker='x')
    plt.xticks(etas, labels=[f'$\eta_{i+1}$' for i, eta in enumerate(etas)])
    plt.xlabel(f'$\eta$')
    plt.ylabel(f'average final regret')
    plt.title(f'{title}')
    if save_fig:
        plt.savefig(f'{save_name}.pdf')
    plt.show()