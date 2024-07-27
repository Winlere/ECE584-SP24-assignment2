import argparse
import matplotlib.pyplot as plt

def know_gap(File):
    with open(File, 'r') as f:
        lines = f.readlines()
        optimized = []
        ids = []
        groudtruth = None
        for line in lines:
            #Minimum y_7 - y0 =    -65.255694; calculated margin =    -14.299698
            if 'Minimum' in line and 'calculated' in line:
                optimized.append(float(line.split()[5][:-1]))
                ids.append(int(line.split()[3][1:]))
                groudtruth = int(line.split()[1][2:])
            if "NOT VALID" in line:
                return None, None, None
        return optimized, ids, groudtruth

def draw_gap(lp_opt, milp_opt, ids, groudtruth, epsilon, ylims):
    # plt.plot(lp_opt, label='LP Optimized', marker='o')
    # plt.plot(milp_opt, label='MILP Optimized (Exact)', marker='o')
    # no line to connect
    plt.plot(lp_opt, label='LP Optimized', marker='o', linestyle='-', linewidth=3)
    if milp_opt is not None:
        plt.plot(milp_opt, label='MILP Optimized (Exact)', marker='o', linestyle='--', linewidth=3)
    else:
        plt.plot([], label='MILP Timeout(10m) (Exact)', marker='o', linestyle='--', linewidth=3)
    plt.legend()
    xlabels = [f'$\\min(y_{groudtruth}-y_{i})$' for i in ids]
    plt.xticks(range(len(ids)), xlabels, rotation=45)
    plt.ylabel('Value')
    plt.ylim(ylims)
    yticks = list(range(-90, 40, 10))
    plt.yticks(yticks)
    # draw y = 0
    plt.axhline(0, color='green', linewidth=1)
    plt.grid()
    plt.title('LP Optimized vs MILP Optimized (s = {})'.format(epsilon))
    plt.tight_layout()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Folder containing the gap files')
    parser.add_argument('epsilon', type=str, help='Comma-separated list of epsilons')
    args = parser.parse_args()
    epsilons = args.epsilon.strip().split(',')
    plt.rcParams['font.serif'] = ['Times New Roman']
    # set font size 
    plt.rc('font', size=12)
    ratio = 2 / 2
    scale = 5
    plt.figure(figsize=(scale * 2 , scale * 3))
    for id,epsilon in enumerate(epsilons):
        lp_opt, ids, groudtruth = know_gap(f'{args.folder}/lp.py.{epsilon}.txt')
        mip_opt, _, _ = know_gap(f'{args.folder}/mip.py.{epsilon}.txt')
        # set to time roman font
        plt.subplot((len(epsilons)+1) // 2, 2, id + 1)
        draw_gap(lp_opt, mip_opt, ids, groudtruth, epsilon, (-85,30))
    # set ratio

    plt.savefig(f'{args.folder}/gap.pdf')