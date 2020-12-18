import time
from util.util import error, print_timestamped, normalize_with_opt
import numpy as np
import os


class ExcelEvaluate:
    def __init__(self, filepath, excel=False):
        self.excel_filename = None
        self.excel = excel
        if self.excel:
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            self.excel_filename = filepath
            self.ff = open(self.excel_filename, "w")
            init_rows = [
                "query_filename",
                "filter",
                "MSE",
                "relMSE",
                "TumourMSE",
                "sMAPE",
                "scaledMSE",
                "scaledrelMSE",
                "scaledTumourMSE",
                "scaledsMAPE",
            ]
            for i, n in enumerate(init_rows):
                self.ff.write(n)
                if i < len(init_rows) - 1:
                    self.ff.write(",")
                else:
                    self.ff.write("\n")

    def print_to_excel(self, data):
        for i, d in enumerate(data):
            self.ff.write(str(d))
            if i < len(data) - 1:
                self.ff.write(",")
            else:
                self.ff.write("\n")

    def evaluate(self, mri_dict, query_name, smoothing="median"):
        r_real = mri_dict['real_B'].flatten()
        r_predicted = mri_dict['fake_B'].flatten()
        r_predicted_smoothed = mri_dict['fake_B_smoothed'].flatten()
        truth_nonzero = None
        if 'truth' in mri_dict:
            r_truth_mri = mri_dict['truth'].flatten()
            truth_nonzero = np.nonzero(r_truth_mri)[0]
            if len(truth_nonzero) == 0:
                truth_nonzero = None
        mse, relmse, tumour, smape = evaluate_result(r_real, r_predicted, tumour_indices=truth_nonzero)
        s_mse, s_relmse, s_tumour, s_smape = evaluate_result(r_real, r_predicted_smoothed,
                                                             tumour_indices=truth_nonzero)

        print_timestamped("MSE of values after normalize")
        r_real = normalize_with_opt(mri_dict['real_B'], 0).flatten()
        r_predicted = normalize_with_opt(mri_dict['fake_B'], 0).flatten()
        r_predicted_smoothed = normalize_with_opt(mri_dict['fake_B_smoothed'], 0).flatten()
        scaledmse, scaledrelmse, scaledtumour, scaledsmape = evaluate_result(r_real,
                                                                             r_predicted,
                                                                             tumour_indices=truth_nonzero)
        scaleds_mse, scaleds_relmse, scaleds_tumour, scaleds_smape = evaluate_result(r_real,
                                                                                     r_predicted_smoothed,
                                                                                     tumour_indices=truth_nonzero)
        if smoothing == "median":
            smoothing = 0
        elif smoothing == "average":
            smoothing = 1
        if self.excel:
            self.print_to_excel([query_name, -1,
                                 mse, relmse, tumour, smape,
                                 scaledmse, scaledrelmse, scaledtumour, scaledsmape,
                                 ])
            self.print_to_excel([query_name, smoothing,
                                 s_mse, s_relmse, s_tumour, s_smape,
                                 scaleds_mse, scaleds_relmse, scaleds_tumour, scaleds_smape
                                 ])

    def close(self):
        if self.excel:
            self.ff.close()
            print_timestamped("Saved in " + str(self.excel_filename))


def common_nonzero(transformed_mris):
    common_nonzero_set = set()
    for i in range(len(transformed_mris)):
        common_nonzero_set = common_nonzero_set.union(set(np.nonzero(transformed_mris[i])[0]))

    return np.array(list(common_nonzero_set))


def evaluate_result(seq, learned_seq, tumour_indices=None, round_fact=6, multiplier=1):
    init = time.time()
    if seq.shape != learned_seq.shape:
        error("The shape of the target and learned sequencing are not the same.")

    tumour = None

    common = common_nonzero([seq, learned_seq])

    mse = np.mean(np.square(seq[common] - learned_seq[common]))
    relmse = mse / np.mean(np.square(seq[common]))
    smape = np.sum(np.abs(seq[common] - learned_seq[common])) / \
            np.sum(np.abs(learned_seq[common]) + np.abs(seq[common]))

    mse = round(mse * multiplier, round_fact)
    print("The mean squared error is " + str(mse) + ".")

    relmse = round(relmse * multiplier, round_fact)
    print("The ratio of MSE and all-zero MSE is " + str(relmse) + ".")

    smape = round(smape, round_fact)
    print("The symmetric mean absolute percentage error is " + str(smape) + ".")

    if tumour_indices is not None:
        tumour = np.mean(np.square(seq[tumour_indices] - learned_seq[tumour_indices]))
        tumour = round(tumour * multiplier, round_fact)
        print("The mean squared error of the tumor is " + str(tumour) + ".")

    end = round(time.time() - init, 3)
    print_timestamped("Time spent computing the error for the current mapping: " + str(end) + "s.")
    return mse, relmse, tumour, smape
