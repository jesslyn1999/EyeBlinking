import json


# with open("./output/video_18000 Tue Sep  8 021847 2020/processed_result.json") \
#         as json_file:
#     input_processed_result = json.load(json_file)
#
#
# def convert(result):
#     out = result
#     # print("HMM:", result)
#     out["idx"] = result.get("idx") - 318
#     return out
#
#
# out_processed_result = list(map(lambda x: convert(x), input_processed_result))
#
#
# with open("./dump.json", "w+") as outfile:
#     json.dump(out_processed_result, outfile)

with open("./dump.json") as json_file:
    input_processed_result_1 = list(json.load(json_file))

with open("./output/video_18000 Mon Sep  7 205352 2020/processed_result.json") as json_file:
    input_processed_result_2 = list(json.load(json_file))

# print(input_processed_result_1[0])
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print(input_processed_result_2[0])
input_processed_result_1.extend(list(input_processed_result_2))

with open("./dump_2.json", "w+") as outfile:
    json.dump(input_processed_result_1, outfile)

print(len(input_processed_result_1))
