"""
- The architectures considered in the thesis distinguishes three types 
of model input for the independent data x:
  (A) Word-embeddings: 		 (#dialogues, #utterances, max_utterance_len)
  (B) Character-embeddings:  (#dialogues, #utterances, max_utterance_len, max_word_len )   
  (C) BERT-embeddings        (#dialogues, #utterances, max_utterance_len) for tokens and segments
	  where - #dialogues & #utterances depending per set/dialogue
			- max_utterance_len & max_word_len are static 
			

- The DA tag data is the same for all model architectures: (#dialogues, #utterances, num_dialogue_acts)
	  where - num_dialogue_acts is dependent per corpus 
"""

# import packages
import pickle
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import operator
import pandas as pd
import numpy as np
import nltk
from itertools import chain
from transformers import BertTokenizer
import os


data = 'SwDA'

# split by Lee & Dernocourt (2016) (arXiv preprint arXiv:1603.03827 .)
if data == 'SwDA':
	train_set_idx = ['sw2005', 'sw2006', 'sw2008', 'sw2010', 'sw2012', 'sw2015', 'sw2018', 'sw2019', 'sw2020', 'sw2022', 'sw2024', 'sw2025', 'sw2027', 'sw2028', 'sw2032', 'sw2035', 'sw2038', 'sw2039', 'sw2040', 'sw2041', 'sw2051', 'sw2060', 'sw2061', 'sw2062', 'sw2064', 'sw2065', 'sw2073', 'sw2078', 'sw2079', 'sw2085', 'sw2086', 'sw2090', 'sw2092', 'sw2093', 'sw2094', 'sw2095', 'sw2101', 'sw2102', 'sw2104', 'sw2105', 'sw2107', 'sw2109', 'sw2110', 'sw2111', 'sw2113', 'sw2120', 'sw2122', 'sw2124', 'sw2125', 'sw2130', 'sw2137', 'sw2139', 'sw2145', 'sw2149', 'sw2154', 'sw2155', 'sw2157', 'sw2168', 'sw2171', 'sw2177', 'sw2178', 'sw2180', 'sw2181', 'sw2184', 'sw2185', 'sw2187', 'sw2190', 'sw2191', 'sw2197', 'sw2205', 'sw2220', 'sw2221', 'sw2226', 'sw2227', 'sw2228', 'sw2231', 'sw2232', 'sw2234', 'sw2235', 'sw2237', 'sw2241', 'sw2244', 'sw2247', 'sw2248', 'sw2249', 'sw2252', 'sw2259', 'sw2260', 'sw2262', 'sw2263', 'sw2264', 'sw2265', 'sw2266', 'sw2268', 'sw2275', 'sw2278', 'sw2279', 'sw2283', 'sw2285', 'sw2287', 'sw2290', 'sw2292', 'sw2293', 'sw2295', 'sw2296', 'sw2300', 'sw2301', 'sw2302', 'sw2303', 'sw2304', 'sw2305', 'sw2308', 'sw2309', 'sw2313', 'sw2314', 'sw2316', 'sw2323', 'sw2324', 'sw2325', 'sw2330', 'sw2331', 'sw2334', 'sw2336', 'sw2339', 'sw2342', 'sw2344', 'sw2349', 'sw2353', 'sw2354', 'sw2355', 'sw2362', 'sw2365', 'sw2366', 'sw2368', 'sw2370', 'sw2372', 'sw2376', 'sw2379', 'sw2380', 'sw2382', 'sw2383', 'sw2386', 'sw2387', 'sw2389', 'sw2393', 'sw2397', 'sw2405', 'sw2406', 'sw2407', 'sw2413', 'sw2418', 'sw2421', 'sw2423', 'sw2424', 'sw2426', 'sw2427', 'sw2429', 'sw2431', 'sw2432', 'sw2433', 'sw2435', 'sw2436', 'sw2437', 'sw2439', 'sw2442', 'sw2445', 'sw2446', 'sw2448', 'sw2450', 'sw2451', 'sw2452', 'sw2457', 'sw2460', 'sw2465', 'sw2466', 'sw2467', 'sw2469', 'sw2471', 'sw2472', 'sw2476', 'sw2477', 'sw2478', 'sw2479', 'sw2482', 'sw2483', 'sw2485', 'sw2486', 'sw2488', 'sw2490', 'sw2492', 'sw2495', 'sw2499', 'sw2502', 'sw2504', 'sw2506', 'sw2510', 'sw2511', 'sw2514', 'sw2515', 'sw2519', 'sw2521', 'sw2524', 'sw2525', 'sw2526', 'sw2527', 'sw2528', 'sw2533', 'sw2537', 'sw2539', 'sw2540', 'sw2543', 'sw2545', 'sw2546', 'sw2547', 'sw2548', 'sw2549', 'sw2552', 'sw2554', 'sw2557', 'sw2559', 'sw2562', 'sw2565', 'sw2566', 'sw2568', 'sw2570', 'sw2571', 'sw2575', 'sw2576', 'sw2578', 'sw2579', 'sw2584', 'sw2585', 'sw2586', 'sw2587', 'sw2589', 'sw2597', 'sw2599', 'sw2602', 'sw2603', 'sw2604', 'sw2608', 'sw2609', 'sw2610', 'sw2611', 'sw2614', 'sw2615', 'sw2616', 'sw2617', 'sw2619', 'sw2622', 'sw2627', 'sw2628', 'sw2631', 'sw2634', 'sw2638', 'sw2640', 'sw2641', 'sw2642', 'sw2645', 'sw2647', 'sw2648', 'sw2650', 'sw2652', 'sw2657', 'sw2658', 'sw2661', 'sw2662', 'sw2663', 'sw2667', 'sw2669', 'sw2672', 'sw2675', 'sw2676', 'sw2678', 'sw2679', 'sw2684', 'sw2689', 'sw2690', 'sw2691', 'sw2692', 'sw2693', 'sw2703', 'sw2707', 'sw2708', 'sw2709', 'sw2710', 'sw2711', 'sw2716', 'sw2717', 'sw2719', 'sw2723', 'sw2726', 'sw2729', 'sw2734', 'sw2736', 'sw2741', 'sw2743', 'sw2744', 'sw2749', 'sw2751', 'sw2754', 'sw2756', 'sw2759', 'sw2761', 'sw2766', 'sw2767', 'sw2768', 'sw2770', 'sw2773', 'sw2774', 'sw2775', 'sw2780', 'sw2782', 'sw2784', 'sw2785', 'sw2788', 'sw2789', 'sw2792', 'sw2793', 'sw2794', 'sw2797', 'sw2800', 'sw2803', 'sw2806', 'sw2812', 'sw2818', 'sw2819', 'sw2820', 'sw2821', 'sw2826', 'sw2827', 'sw2828', 'sw2830', 'sw2834', 'sw2835', 'sw2837', 'sw2840', 'sw2844', 'sw2847', 'sw2849', 'sw2851', 'sw2858', 'sw2860', 'sw2862', 'sw2866', 'sw2868', 'sw2870', 'sw2871', 'sw2875', 'sw2876', 'sw2877', 'sw2879', 'sw2883', 'sw2884', 'sw2887', 'sw2893', 'sw2896', 'sw2897', 'sw2898', 'sw2900', 'sw2909', 'sw2910', 'sw2913', 'sw2915', 'sw2917', 'sw2921', 'sw2924', 'sw2926', 'sw2927', 'sw2929', 'sw2930', 'sw2932', 'sw2934', 'sw2935', 'sw2938', 'sw2942', 'sw2945', 'sw2950', 'sw2952', 'sw2953', 'sw2954', 'sw2955', 'sw2956', 'sw2957', 'sw2960', 'sw2962', 'sw2963', 'sw2965', 'sw2967', 'sw2968', 'sw2969', 'sw2970', 'sw2982', 'sw2983', 'sw2984', 'sw2991', 'sw2992', 'sw2993', 'sw2994', 'sw2995', 'sw2996', 'sw2998', 'sw2999', 'sw3000', 'sw3001', 'sw3002', 'sw3003', 'sw3004', 'sw3007', 'sw3009', 'sw3011', 'sw3012', 'sw3013', 'sw3014', 'sw3016', 'sw3018', 'sw3019', 'sw3020', 'sw3021', 'sw3023', 'sw3025', 'sw3028', 'sw3029', 'sw3030', 'sw3034', 'sw3036', 'sw3038', 'sw3039', 'sw3040', 'sw3041', 'sw3042', 'sw3045', 'sw3047', 'sw3049', 'sw3050', 'sw3051', 'sw3052', 'sw3054', 'sw3055', 'sw3056', 'sw3057', 'sw3059', 'sw3061', 'sw3062', 'sw3063', 'sw3064', 'sw3065', 'sw3067', 'sw3068', 'sw3069', 'sw3070', 'sw3071', 'sw3073', 'sw3074', 'sw3075', 'sw3076', 'sw3077', 'sw3080', 'sw3081', 'sw3082', 'sw3083', 'sw3085', 'sw3086', 'sw3087', 'sw3088', 'sw3090', 'sw3092', 'sw3093', 'sw3095', 'sw3097', 'sw3099', 'sw3102', 'sw3103', 'sw3104', 'sw3105', 'sw3107', 'sw3108', 'sw3111', 'sw3113', 'sw3115', 'sw3118', 'sw3120', 'sw3121', 'sw3124', 'sw3130', 'sw3131', 'sw3133', 'sw3134', 'sw3135', 'sw3136', 'sw3138', 'sw3140', 'sw3142', 'sw3143', 'sw3144', 'sw3146', 'sw3150', 'sw3151', 'sw3152', 'sw3154', 'sw3155', 'sw3158', 'sw3159', 'sw3161', 'sw3162', 'sw3166', 'sw3167', 'sw3168', 'sw3169', 'sw3170', 'sw3171', 'sw3173', 'sw3174', 'sw3175', 'sw3182', 'sw3185', 'sw3186', 'sw3187', 'sw3188', 'sw3189', 'sw3194', 'sw3195', 'sw3196', 'sw3198', 'sw3200', 'sw3201', 'sw3203', 'sw3204', 'sw3205', 'sw3206', 'sw3208', 'sw3214', 'sw3215', 'sw3216', 'sw3219', 'sw3221', 'sw3223', 'sw3225', 'sw3226', 'sw3227', 'sw3228', 'sw3229', 'sw3230', 'sw3231', 'sw3232', 'sw3233', 'sw3234', 'sw3235', 'sw3236', 'sw3237', 'sw3238', 'sw3242', 'sw3244', 'sw3245', 'sw3247', 'sw3252', 'sw3253', 'sw3254', 'sw3256', 'sw3259', 'sw3260', 'sw3265', 'sw3266', 'sw3267', 'sw3268', 'sw3269', 'sw3270', 'sw3271', 'sw3272', 'sw3275', 'sw3276', 'sw3279', 'sw3280', 'sw3282', 'sw3283', 'sw3284', 'sw3286', 'sw3293', 'sw3294', 'sw3296', 'sw3300', 'sw3303', 'sw3304', 'sw3306', 'sw3309', 'sw3310', 'sw3311', 'sw3313', 'sw3315', 'sw3317', 'sw3319', 'sw3320', 'sw3324', 'sw3325', 'sw3326', 'sw3327', 'sw3328', 'sw3330', 'sw3331', 'sw3332', 'sw3333', 'sw3338', 'sw3340', 'sw3342', 'sw3343', 'sw3344', 'sw3345', 'sw3349', 'sw3351', 'sw3353', 'sw3355', 'sw3359', 'sw3360', 'sw3361', 'sw3362', 'sw3363', 'sw3364', 'sw3365', 'sw3367', 'sw3368', 'sw3369', 'sw3371', 'sw3372', 'sw3373', 'sw3375', 'sw3377', 'sw3379', 'sw3381', 'sw3383', 'sw3384', 'sw3386', 'sw3387', 'sw3389', 'sw3393', 'sw3397', 'sw3398', 'sw3399', 'sw3402', 'sw3403', 'sw3405', 'sw3406', 'sw3408', 'sw3409', 'sw3411', 'sw3414', 'sw3417', 'sw3419', 'sw3420', 'sw3421', 'sw3424', 'sw3425', 'sw3426', 'sw3427', 'sw3428', 'sw3429', 'sw3431', 'sw3435', 'sw3439', 'sw3441', 'sw3443', 'sw3447', 'sw3448', 'sw3449', 'sw3450', 'sw3451', 'sw3453', 'sw3454', 'sw3455', 'sw3457', 'sw3458', 'sw3460', 'sw3463', 'sw3464', 'sw3467', 'sw3473', 'sw3476', 'sw3487', 'sw3489', 'sw3495', 'sw3496', 'sw3503', 'sw3504', 'sw3508', 'sw3513', 'sw3514', 'sw3515', 'sw3517', 'sw3518', 'sw3521', 'sw3523', 'sw3524', 'sw3525', 'sw3526', 'sw3527', 'sw3530', 'sw3533', 'sw3535', 'sw3537', 'sw3539', 'sw3541', 'sw3543', 'sw3549', 'sw3550', 'sw3551', 'sw3556', 'sw3557', 'sw3561', 'sw3563', 'sw3565', 'sw3567', 'sw3569', 'sw3570', 'sw3573', 'sw3574', 'sw3580', 'sw3586', 'sw3591', 'sw3595', 'sw3596', 'sw3597', 'sw3606', 'sw3607', 'sw3615', 'sw3624', 'sw3626', 'sw3628', 'sw3633', 'sw3636', 'sw3638', 'sw3639', 'sw3642', 'sw3646', 'sw3647', 'sw3651', 'sw3655', 'sw3657', 'sw3660', 'sw3662', 'sw3663', 'sw3665', 'sw3676', 'sw3680', 'sw3681', 'sw3682', 'sw3688', 'sw3691', 'sw3692', 'sw3693', 'sw3694', 'sw3696', 'sw3699', 'sw3703', 'sw3707', 'sw3709', 'sw3716', 'sw3720', 'sw3723', 'sw3725', 'sw3727', 'sw3728', 'sw3734', 'sw3735', 'sw3736', 'sw3738', 'sw3743', 'sw3745', 'sw3746', 'sw3747', 'sw3750', 'sw3751', 'sw3754', 'sw3760', 'sw3763', 'sw3764', 'sw3768', 'sw3770', 'sw3773', 'sw3774', 'sw3776', 'sw3777', 'sw3781', 'sw3784', 'sw3788', 'sw3791', 'sw3796', 'sw3798', 'sw3801', 'sw3802', 'sw3803', 'sw3804', 'sw3805', 'sw3809', 'sw3813', 'sw3815', 'sw3821', 'sw3825', 'sw3828', 'sw3830', 'sw3838', 'sw3841', 'sw3845', 'sw3847', 'sw3850', 'sw3852', 'sw3855', 'sw3862', 'sw3870', 'sw3876', 'sw3883', 'sw3887', 'sw3898', 'sw3902', 'sw3903', 'sw3908', 'sw3911', 'sw3917', 'sw3925', 'sw3926', 'sw3946', 'sw3952', 'sw3956', 'sw3962', 'sw3965', 'sw3971', 'sw3979', 'sw3983', 'sw3985', 'sw3988', 'sw3993', 'sw4008', 'sw4013', 'sw4019', 'sw4022', 'sw4023', 'sw4028', 'sw4032', 'sw4033', 'sw4036', 'sw4038', 'sw4049', 'sw4050', 'sw4051', 'sw4055', 'sw4056', 'sw4060', 'sw4064', 'sw4071', 'sw4074', 'sw4077', 'sw4078', 'sw4079', 'sw4080', 'sw4082', 'sw4090', 'sw4092', 'sw4096', 'sw4099', 'sw4101', 'sw4103', 'sw4104', 'sw4108', 'sw4109', 'sw4113', 'sw4114', 'sw4123', 'sw4127', 'sw4129', 'sw4130', 'sw4133', 'sw4137', 'sw4138', 'sw4147', 'sw4148', 'sw4149', 'sw4150', 'sw4151', 'sw4152', 'sw4153', 'sw4154', 'sw4155', 'sw4158', 'sw4159', 'sw4165', 'sw4166', 'sw4168', 'sw4171', 'sw4174', 'sw4175', 'sw4177', 'sw4181', 'sw4184', 'sw4311', 'sw4312', 'sw4314', 'sw4316', 'sw4319', 'sw4320', 'sw4325', 'sw4327', 'sw4329', 'sw4330', 'sw4333', 'sw4334', 'sw4336', 'sw4339', 'sw4340', 'sw4341', 'sw4342', 'sw4345', 'sw4346', 'sw4349', 'sw4353', 'sw4358', 'sw4360', 'sw4362', 'sw4363', 'sw4364', 'sw4366', 'sw4370', 'sw4376', 'sw4378', 'sw4379', 'sw4380', 'sw4382', 'sw4443', 'sw4483', 'sw4519', 'sw4548', 'sw4565', 'sw4603', 'sw4605', 'sw4608', 'sw4611', 'sw4615', 'sw4617', 'sw4618', 'sw4619', 'sw4626', 'sw4628', 'sw4630', 'sw4642', 'sw4644', 'sw4646', 'sw4649', 'sw4655', 'sw4659', 'sw4666', 'sw4675', 'sw4679', 'sw4681', 'sw4682', 'sw4688', 'sw4691', 'sw4698', 'sw4703', 'sw4709', 'sw4720', 'sw4721', 'sw4723', 'sw4725', 'sw4726', 'sw4728', 'sw4733', 'sw4735', 'sw4745', 'sw4752', 'sw4758', 'sw4759', 'sw4765', 'sw4770', 'sw4774', 'sw4784', 'sw4785', 'sw4788', 'sw4792', 'sw4796', 'sw4799', 'sw4801', 'sw4812', 'sw4814', 'sw4821', 'sw4822', 'sw4826', 'sw4829', 'sw4830', 'sw4831', 'sw4834', 'sw4840', 'sw4856', 'sw4858', 'sw4859', 'sw4868', 'sw4876', 'sw4877', 'sw4880', 'sw4886', 'sw4902', 'sw4905', 'sw4908', 'sw4927', 'sw4928', 'sw4936', 'sw4940']
	valid_set_idx = ['sw2053', 'sw2067', 'sw2071', 'sw2072', 'sw2160', 'sw2163', 'sw2175', 'sw2253', 'sw2289', 'sw2299', 'sw2340', 'sw2373', 'sw2395', 'sw2399', 'sw2455', 'sw2501', 'sw2534', 'sw2558', 'sw2593', 'sw2594', 'sw2598', 'sw2620', 'sw2621', 'sw2623', 'sw2630', 'sw2653', 'sw2713', 'sw2755', 'sw2772', 'sw2776', 'sw2790', 'sw2832', 'sw2839', 'sw2842', 'sw2854', 'sw2874', 'sw2888', 'sw2889', 'sw2944', 'sw2959', 'sw2981', 'sw2989', 'sw3015', 'sw3046', 'sw3072', 'sw3096', 'sw3148', 'sw3156', 'sw3181', 'sw3184', 'sw3190', 'sw3191', 'sw3202', 'sw3207', 'sw3239', 'sw3246', 'sw3250', 'sw3251', 'sw3255', 'sw3257', 'sw3281', 'sw3288', 'sw3290', 'sw3291', 'sw3334', 'sw3346', 'sw3352', 'sw3354', 'sw3382', 'sw3433', 'sw3445', 'sw3491', 'sw3497', 'sw3500', 'sw3506', 'sw3509', 'sw3554', 'sw3576', 'sw3584', 'sw3587', 'sw3658', 'sw3659', 'sw3666', 'sw3675', 'sw3686', 'sw3697', 'sw3711', 'sw3769', 'sw3797', 'sw3810', 'sw3811', 'sw3921', 'sw4004', 'sw4026', 'sw4037', 'sw4048', 'sw4072', 'sw4318', 'sw4321', 'sw4347', 'sw4356', 'sw4372', 'sw4572', 'sw4633', 'sw4660', 'sw4697', 'sw4707', 'sw4716', 'sw4736', 'sw4802', 'sw4890', 'sw4917']
	test_set_idx = ['sw2121', 'sw2131', 'sw2151', 'sw2229', 'sw2335', 'sw2434', 'sw2441', 'sw2461', 'sw2503', 'sw2632', 'sw2724', 'sw2752', 'sw2753', 'sw2836', 'sw2838', 'sw3528', 'sw3756', 'sw3942', 'sw3994']

	# data directories to load data
	data_dir = "data\\raw_data\\SwDA\\"
	processed_data_dir = "data\\processed_data\\SwDA\\"
elif data == 'MRDA':
	train_set_idx = ['Bdb001', 'Bed002', 'Bed004', 'Bed005', 'Bed008', 'Bed009', 'Bed011', 'Bed013', 'Bed014', 'Bed015', 'Bed017', 'Bmr002', 'Bmr003', 'Bmr006', 'Bmr007', 'Bmr008', 'Bmr009', 'Bmr011', 'Bmr012', 'Bmr015', 'Bmr016', 'Bmr020', 'Bmr021', 'Bmr023', 'Bmr025', 'Bmr026', 'Bmr027', 'Bmr029', 'Bmr031', 'Bns001', 'Bns002', 'Bns003', 'Bro003', 'Bro005', 'Bro007', 'Bro010', 'Bro012', 'Bro013', 'Bro015', 'Bro016', 'Bro017', 'Bro019', 'Bro022', 'Bro023', 'Bro025', 'Bro026', 'Bro028', 'Bsr001', 'Btr001', 'Btr002', 'Buw001']
    valid_set_idx = ['Bed003', 'Bed010', 'Bmr005', 'Bmr014', 'Bmr019', 'Bmr024', 'Bmr030', 'Bro004', 'Bro011', 'Bro018', 'Bro024']
    test_set_idx =  ['Bed006', 'Bed012', 'Bed016', 'Bmr001', 'Bmr010', 'Bmr022', 'Bmr028', 'Bro008', 'Bro014', 'Bro021', 'Bro027']
    data_dir = "data\\raw_data\\MRDA\\"
    processed_data_dir = "data\\processed_data\\MRDA\\"


# load metadata
metadata = pickle.load(open(processed_data_dir + "metadata.pkl", "rb"))


#------------------ (A) input for  Word-embeddings ---------------------#

def get_utterances_tags(data):
	"""
	This function returns a list with all tokenized 
	utterances and da_tags for the specified data set
	"""
	utterances = []
	da_tags = []

	data['utterance'] = data['utterance'].str.lower()

	for i in range(0,len(data)):
		utterance = data.iloc[i,1]
		da = data.iloc[i,2]
		if np.str(utterance) != 'nan':
			current_utterance = nltk.word_tokenize(utterance)
			utterances.append(current_utterance)
			da_tags.append(da)
		else:
			continue
			
	return utterances, da_tags

def generate_embeddings(data, metadata):
	"""
	This function returns for each utterance in the
	data set a list with word-features (i.e. the word
	index on the word position) and a list with da_tag
	features (i.e. the tag label on the label position)
	"""
	# load parameters
	num_tags = metadata['num_da_tags']
	number_of_utterances = metadata['num_utterances']
	word_to_index = metadata['word_to_index']
	label_to_index = metadata['label_to_index']
	max_utterance_len = metadata['max_utterance_len']
	
	# get utterances and tags
	utterances = data[0]
	tags = data[1]
	
	tmp_utterance_embeddings = []
	tmp_label_embeddings = []

	# Convert each word and label into its numerical representation
	for i in range(len(utterances)):
		tmp_utt = []
		for word in utterances[i]:
			if word in word_to_index:
				tmp_utt.append(word_to_index[word])
			else:
				tmp_utt.append(0)

		tmp_utterance_embeddings.append(tmp_utt)
		tmp_label_embeddings.append(label_to_index[tags[i]])

	# For Keras LSTM must pad the sequences to same length and return a numpy array
	utterance_embeddings = pad_sequences(tmp_utterance_embeddings, maxlen=max_utterance_len, padding='post', value=0.0)

	# Convert labels to one hot vectors
	label_embeddings = to_categorical(np.asarray(tmp_label_embeddings), num_classes=num_tags)
	
	return utterance_embeddings, label_embeddings

def get_data(data_dir, set_idx):
	data_list_x = []
	data_list_y = []
	for i in range(0,len(set_idx)):
		new_data = pd.read_csv(data_dir + set_idx[i].split('w')[1] + '.txt', sep = '|', header = None, names = ['speaker', 'utterance', 'DA'])

		data_process = get_utterances_tags(new_data)
		pre_data_x, pre_data_y = generate_embeddings(data_process, metadata)

		data_list_x.append(pre_data_x)
		data_list_y.append(pre_data_y)

	data_x = np.array(([data_list_x[i] for i in range(0,len(data_list_x))]))
	data_y = np.array(([data_list_y[i] for i in range(0,len(data_list_y))]))
	
	return data_x, data_y


print('-------- storing word-embedding data ----------')
# save data
train_x, train_y = get_data(data_dir, train_set_idx)
pickle.dump(train_x, open(processed_data_dir + 'train_x.pkl', 'wb'))
pickle.dump(train_y, open(processed_data_dir + 'train_y.pkl', 'wb'))

test_x, test_y = get_data(data_dir, test_set_idx)
pickle.dump(test_x, open(processed_data_dir + 'test_x.pkl', 'wb'))
pickle.dump(test_y, open(processed_data_dir + 'test_y.pkl', 'wb'))

valid_x, valid_y =  get_data(data_dir, valid_set_idx)
pickle.dump(valid_x, open(processed_data_dir + 'valid_x.pkl', 'wb'))
pickle.dump(valid_y, open(processed_data_dir + 'valid_y.pkl', 'wb'))


#------------------ (B) input for Character-embeddings ---------------------#
word_to_index = metadata['word_to_index']
index_to_word = metadata['index_to_word']
vocabulary_size = metadata['vocabulary_size']

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
	char_dict[char] = i + 1
	
char_dict['UNK'] = 69   # add unknown token to represent rare characters in vocabulary

alphabet_size = len(char_dict)

max_word_len = 0
for word, i in word_to_index.items():
	if len(word) > max_word_len:
		max_word_len = len(word)
		
total_char_features = []
for word, i in word_to_index.items():
	char_feat = []
	for j, char in enumerate(word):
		char_feat.append(char_dict[char])
	
	total_char_features.append(char_feat)
	
char_embedding = pad_sequences(total_char_features, maxlen=max_word_len, padding='post', value=0.0)


def char_embd_data(data, char_dict, index_to_word):	
	"""
	This function returns the character-embedding 
	format
	"""
	char_embd = []
	for x in range(0,len(data)):
		char_tmp1 = []
		for z in range(0,len(data[x])):
			char_tmp2 = []
			char_tmp3 = []
			for i in range(0,131):
				char_feat = []
				if data[x][z][i] != 0:
					for j, char in enumerate(index_to_word[data[x][z][i]]):
						if char in char_dict:
							char_feat.append(char_dict[char])
						else:
							char_feat.append(char_dict['UNK'])

					char_tmp2.append(char_feat)
				else:
					char_tmp2.append([0])

			char_tmp3.append(char_tmp2)

			char_inner = pad_sequences(list(chain(*char_tmp3)), maxlen=max_word_len, padding='post', value=0.0)
			char_tmp1.append(char_inner)

		char_combined = np.array(([char_tmp1[i] for i in range(0,len(char_tmp1))]))
		char_embd.append(char_combined)
		
	data_char_x = np.array(([char_embd[i] for i in range(0,len(char_embd))]))

	return data_char_x


print('-------- storing character-embedding data ----------')
# save data
train_char_x = char_embd_data(train_x, char_dict, index_to_word)
valid_char_x = char_embd_data(valid_x, char_dict, index_to_word)
test_char_x = char_embd_data(test_x, char_dict, index_to_word)

pickle.dump(train_char_x, open(processed_data_dir + 'train_char_x.pkl', 'wb'))
pickle.dump(valid_char_x, open(processed_data_dir + 'valid_char_x.pkl', 'wb'))
pickle.dump(test_char_x, open(processed_data_dir + 'test_char_x.pkl', 'wb'))


#------------------ (C) input for Bert-embeddings ---------------------#

# load BERt pre-trained embeddings
bert_pretrained_dir = 'embeddings\\pre-trained\\uncased_L-12_H-768_A-12'
config_path = os.path.join(bert_pretrained_dir, 'bert_config.json')
checkpoint_path = os.path.join(bert_pretrained_dir, 'bert_model.ckpt')
vocab_path = os.path.join(bert_pretrained_dir, 'vocab.txt')

def get_input_and_mask(new_data, metadata, vocab_path):
	"""
	Create token and segment embeddings for BERT
	"""
	
	max_utterance_len = metadata['max_utterance_len']

	# get the list of utterances and their tags
	utterances = new_data.utterance.values
	
	from keras_bert import Tokenizer
	import codecs
	token_dict = {}
	with codecs.open(vocab_path, 'r', 'utf8') as reader:
		for line in reader:
			token = line.strip()
			token_dict[token] = len(token_dict)
	
	tokenizer = Tokenizer(token_dict)
	
	# Tokenize all of the sentences and map the tokens to thier word IDs.
	input_tokens = []
	input_segments = []

	i = 0
	for utt in utterances:
		
		if np.str(utt) != 'nan': 
			input_tokens.append(tokenizer.encode(first=utt, max_len = max_utterance_len+2)[0])
			input_segments.append(tokenizer.encode(first=utt, max_len = max_utterance_len+2)[1])  
		else:
			continue

	input_tokens = np.array(input_tokens)
	input_segments = np.array(input_segments)
	
	return input_tokens, input_segments

def get_bert_data(data_dir, set_idx, metadata, vocab_path):
	"""
	Combine all token and segment embeddings of branch
	"""

	data_list_tokens = []
	data_list_segments = []
	for i in range(0,len(set_idx)): 
		new_data = pd.read_csv(data_dir + set_idx[i].split('w')[1] + '.txt', sep = '|', header = None, names = ['speaker', 'utterance', 'DA'])

		pre_tokens, pre_segments = get_input_and_mask(new_data, metadata, vocab_path)

		data_list_tokens.append(pre_tokens)
		data_list_segments.append(pre_segments)

	data_tokens = np.array(([data_list_tokens[i] for i in range(0,len(data_list_tokens))]))
	data_segments = np.array(([data_list_segments[i] for i in range(0,len(data_list_segments))]))
	
	return data_tokens, data_segments


print('-------- storing BERT-embedding data ----------')
# save data
train_data_token, train_data_segment = get_bert_data(data_dir, train_set_idx, metadata, vocab_path)
valid_data_token, valid_data_segment = get_bert_data(data_dir, valid_set_idx, metadata, vocab_path)
test_data_token, test_data_segment = get_bert_data(data_dir, test_set_idx, metadata, vocab_path)

pickle.dump(train_data_token, open(processed_data_dir + 'train_bert_tokens.pkl', 'wb'))
pickle.dump(train_data_segment, open(processed_data_dir + 'train_bert_segments.pkl', 'wb'))
pickle.dump(valid_data_token, open(processed_data_dir + 'valid_bert_tokens.pkl', 'wb'))
pickle.dump(valid_data_segment, open(processed_data_dir + 'valid_bert_segments.pkl', 'wb'))
pickle.dump(test_data_token, open(processed_data_dir + 'test_bert_tokens.pkl', 'wb'))
pickle.dump(test_data_segment, open(processed_data_dir + 'test_bert_segments.pkl', 'wb'))
