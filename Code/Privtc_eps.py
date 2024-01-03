import numpy as np
import random
import parameter_setting as para
import algorithm_PrivGR_PrivSL
import utility_metric_query_avae as UMquery_avae
import utility_metric_query_FP as UM_FP
import utility_metric_query_length_error as UMlength_error
import matplotlib.pyplot as plt
import time

starttime = time.time()
def setup_args(args=None):
    args.user_num = 5000  # the total number of users
    args.trajectory_len = 6  # the trajectory length
    args.x_left = -45.0  # set the scope of the 2-D geospatial domain
    args.x_right = 85.0
    args.y_left = -160.0
    args.y_right = 160.0

    args.sigma = 0.2  # the parameters in guideline
    args.alpha = 0.02

    args.epsilon = 1.0  # set the privacy budget

    args.query_region_num = 225  # 15 * 15 grid for Query MAE
    args.FP_granularity = 15  # 15 * 15 grid for FP Similarity
    args.length_bin_num = 20  # 20 bins for Distance Error

def load_dataset(txt_dataset_path=None):
    user_record = []
    with open(txt_dataset_path, "r") as fr:
        for line in fr:
            line = line.strip()
            line = line.split()
            user_record.append(line)
    return user_record

def sys_test(epsilon_values):
    txt_dataset_path = "/content/PrivTC-Implementation/Dataset/test_dataset_users_5000_tralen_6.txt" #change dataset location if any error 
    args = para.generate_args()  # define the parameters
    setup_args(args=args)  # set the parameters
    user_record = load_dataset(txt_dataset_path=txt_dataset_path)  # read user data
    random_seed = 1
    random.seed(random_seed)
    np.random.seed(seed=random_seed)
    np.random.shuffle(user_record)

    # generate query *********************************************
    # Query MAE
    query_mae = UMquery_avae.QueryAvAE(args=args)
    query_mae.generated_query_region_list()
    query_mae.real_ans_list = query_mae.get_ans_list_for_each_query_region(user_record=user_record)

    # Distance Error
    distance_error = UMlength_error.LengthError(args=args)
    distance_error.real_length_list = distance_error.get_length_list(user_record=user_record)

    # FP similarity
    fp_similarity = UM_FP.FP(args=args)
    fp_similarity.construct_grid()
    fp_similarity.real_pattern_sorted_dict = fp_similarity.get_pattern_sorted_dict_with_uniform_grid(user_record=user_record)

    # Lists to store results for each epsilon
    mae_results = []
    fp_results = []
    dis_results = []

    for epsilon in epsilon_values:
        args.epsilon = epsilon

        # invoke PrivTC ****************************************************************
        print(f"\nPrivTC starts for Epsilon={epsilon}...")
        algo = algorithm_PrivGR_PrivSL.AG_PrivGR_PrivSL(args=args)
        algo.divide_user_record(user_record)
        algo.construct_grid()
        algo.spectral_learning()
        algo.synthesize_data()
        algorithm_user_record = algo.generated_float_user_record_with_level_2_grid
        print("PrivTC ends!")

        # Query MAE
        algorithm_ans_list = query_mae.get_ans_list_for_each_query_region(user_record=algorithm_user_record)
        mae_ans = query_mae.get_answer_of_metric(ans_real_list=query_mae.real_ans_list, ans_syn_list=algorithm_ans_list)
        print(f"Query MAE (Epsilon={epsilon}): {mae_ans}")
        mae_results.append(mae_ans)

        # FP_similarity
        syn_pattern_sorted_dict = fp_similarity.get_pattern_sorted_dict_with_uniform_grid(user_record=algorithm_user_record)
        fp_ans = fp_similarity.get_FP_similarity(syn_pattern_sorted_dict)
        print(f"FP Similarity (Epsilon={epsilon}): {fp_ans}")
        fp_results.append(fp_ans)

        # Distance Error
        syn_length_list = distance_error.get_length_list(user_record=algorithm_user_record)
        dis_ans = distance_error.get_length_error(syn_length_list)
        print(f"Distance Error (Epsilon={epsilon}): {dis_ans}")
        dis_results.append(dis_ans)

    return mae_results, fp_results, dis_results

if __name__ == '__main__':
    epsilon_values = np.arange(0.5, 5.1, 0.5)   # Add the epsilon values you want to test
    mae_results, fp_results, dis_results = sys_test(epsilon_values)

    # Plotting results
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(epsilon_values, mae_results, marker='o')
    plt.title('Query MAE Results')
    plt.xlabel('Epsilon')
    plt.ylabel('MAE')

    plt.subplot(3, 1, 2)
    plt.plot(epsilon_values, fp_results, marker='o')
    plt.title('FP Similarity Results')
    plt.xlabel('Epsilon')
    plt.ylabel('FP Similarity')

    plt.subplot(3, 1, 3)
    plt.plot(epsilon_values, dis_results, marker='o')
    plt.title('Distance Error Results')
    plt.xlabel('Epsilon')
    plt.ylabel('Distance Error')

    plt.tight_layout()
    plt.show()

print('That took {} seconds'.format(time.time() - starttime))
