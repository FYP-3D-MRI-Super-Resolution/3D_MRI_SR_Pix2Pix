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
                "sMAPE",
                "TumourMSE",
                "scaled_MSE",
                "scaled_relMSE",
                "scaled_sMAPE",
                "scaled_TumourMSE"
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
        mse, relmse, smape, tumour = evaluate_result(mri_dict['real_B'],
                                                     mri_dict['fake_B'],
                                                     tumor_mask=mri_dict['truth'])
        smooth_mse, smooth_relmse, smooth_smape, smooth_tumour = evaluate_result(mri_dict['real_B'],
                                                                                 mri_dict['fake_B_smoothed'],
                                                                                 tumor_mask=mri_dict['truth'])
        print_timestamped("Computing MSE on the scaled data")
        # Scale data in 0,1 and compute everything again
        s_real = normalize_with_opt(mri_dict['real_B'], 0)
        s_predicted = normalize_with_opt(mri_dict['fake_B'], 0)
        s_predicted_smoothed = normalize_with_opt(mri_dict['fake_B_smoothed'], 0)
        scaled_mse, scaled_relmse, scaled_smape, scaled_tumour = evaluate_result(s_real,
                                                                                 s_predicted,
                                                                                 tumor_mask=mri_dict['truth'])
        s_smooth_mse, s_smooth_relmse, s_smooth_smape, s_smooth_tumour = evaluate_result(s_real,
                                                                                         s_predicted_smoothed,
                                                                                         tumor_mask=mri_dict['truth'])
        smoothing = 0 if smoothing == "median" else 1
        if self.excel:
            self.print_to_excel([query_name, -1,
                                 mse, relmse, smape, tumour,
                                 scaled_mse, scaled_relmse, scaled_smape, scaled_tumour,
                                 ])
            self.print_to_excel([query_name, smoothing,
                                 smooth_mse, smooth_relmse, smooth_tumour, smooth_smape,
                                 s_smooth_mse, s_smooth_relmse, s_smooth_tumour, s_smooth_smape,
                                 ])

    def close(self):
        if self.excel:
            self.ff.close()
            print_timestamped("Saved in " + str(self.excel_filename))


def evaluate_result(seq, learned_seq, tumor_mask=None, round_fact=6, multiplier=1):
    if seq.shape != learned_seq.shape:
        error("The shape of the target and learned sequencing are not the same.")

    tumour = None
    mask = (seq != seq.min()) * (learned_seq != learned_seq.min())
    ground_truth = seq[mask]
    prediction = learned_seq[mask]

    # MSE: avg((A-B)^2)
    mse = (np.square(np.subtract(ground_truth, prediction))).mean()
    # RelMSE
    relmse = mse / np.square(ground_truth).mean()
    # SMAPE = sum |F - A| / sum |A| + |F|
    smape = (100 / np.size(ground_truth)) * \
            np.sum(np.abs(ground_truth - prediction) * 2 / (np.abs(prediction) + np.abs(ground_truth)))

    mse = round(mse * multiplier, round_fact)
    print("The mean squared error is " + str(mse) + ".")

    relmse = round(relmse * multiplier, round_fact)
    print("The ratio of MSE and all-zero MSE is " + str(relmse) + ".")

    smape = round(smape, round_fact)
    print("The symmetric mean absolute percentage error is " + str(smape) + ".")

    if tumor_mask is not None:
        tumour = (np.square(np.subtract(seq[tumor_mask], learned_seq[tumor_mask]))).mean()
        tumour = round(tumour * multiplier, round_fact)
        print("The mean squared error of the tumor is " + str(tumour) + ".")

    return mse, relmse, smape, tumour
