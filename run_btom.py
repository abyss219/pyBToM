import numpy as np
from btom import btom_solve_momdps


def run_btom():
    beta_score_values = np.arange(0.5, 10.5, 0.5)

    for beta_score in beta_score_values:
        pass

    btom_solve_momdps(0.5)



run_btom()