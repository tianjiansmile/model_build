import json

feature_name_json = json.load(open('EXTEND_INFO_G2_FEATURE.json', mode="r", encoding='utf-8'))

new_name_json = {}
for key in feature_name_json:
    if 'sar_region' not in key and 'phoneNumLoc' not in key \
            and 'tanzhi' not in key :
        new_name_json[key] = -1

print(new_name_json)

print('new len',len(new_name_json))
with open("EXTEND_INFO_G2_FEATURE_PRO.json","w") as f:
     json.dump(new_name_json,f)
     print("加载入文件完成...")
