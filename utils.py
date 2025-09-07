import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 파일 읽어 리스트로 반환
def read_as_list(x):
    if "'" in x:
        x = x.replace("'", "")
    net = x.rstrip("]").lstrip("[")
    if len(net)==0:
        return None
    return net.split(", ")

# 나이 그룹화
def categorize_age(age):
    if age < 20:
        return 2
    elif 20 <= age <= 29:
        return 3
    elif 30 <= age <= 39:
        return 4
    elif 40 <= age <= 49:
        return 5
    elif 50 <= age <= 59:
        return 6
    elif 60 <= age <= 69:
        return 7
    elif 70 <= age <= 79:
        return 8
    else:
        return 9

# 파일 읽기
def read_input_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# 입력 파일 파싱
def parse_input_text(input_text):
    age = None
    sex = None
    rac = "other/unknown"
    pres_list = []
        
    lines = [each_line.lower() for each_line in input_text.strip().split('\n')]
    for line in lines:
        # Extract age
        if line.startswith('age:'):
            age = int(line.split(':')[1].strip())
        
        # Extract sex
        elif line.startswith('sex:'):
            sex = line.split(':')[1].strip()
            if sex not in ['male', 'female']:
                raise ValueError('Sex should be Male or Female. Current value is %s'%(sex))
        
        # Extract race
        elif line.startswith('race:'):
            race = line.split(':')[1].strip()
            if race in ['white', 'black', 'asian', 'hispanic/latino']:
                rac = race
            elif race in ['hispanic', 'latino']:
                rac = 'hispanic/latino'
            else:
                rac = "other"
        
        # Extract prescriptions
        elif line.startswith('prescriptions:'):
            pres_index = lines.index(line) + 1
            pres_list = [lines[i].strip() for i in range(pres_index, len(lines)) if lines[i].strip()]

    return age, sex, rac, pres_list

# 특징 생성
def generate_feature(input_text_path, feature_type):
    # 입력 데이터 가져오기
    input_text = read_input_file(input_text_path)
    age, sex, rac, pres_list = parse_input_text(input_text)

    # 약물별 특징 정보 가져오기
    ingredients_df = pd.read_csv('./data/ingredients.csv', converters={'Fingerprint':read_as_list, 'DTI':read_as_list})
    # ex {"sevelamer": 12345, "furosemide": 67890, ...}
    rxcui_dict = dict(zip(ingredients_df['Name'].str.lower(), ingredients_df['RxCUI']))
    # rxcui 코드로 인덱스 설정
    ingredients_df = ingredients_df.set_index('RxCUI', drop=True)
    # 피쳐 형태 불러오기
    syn_mimic = pd.read_csv('./data/synthetic_mimic.csv', index_col=0)

    # 피쳐 초기화
    feature_dict = {}
    for each_column in syn_mimic.columns:
        feature_dict[each_column] = 0

    # 환자 특성 피쳐 처리
    feature_dict['Sex_%s'%(sex.upper())] = 1
    feature_dict['Race_%s'%(rac.upper())] = 1
    feature_dict['Age'] = age

    # 처방 데이터 특성 처리
    for each_drug in pres_list:
        if each_drug in rxcui_dict:
            rxcui = str(rxcui_dict[each_drug])
            feature_dict[rxcui] += 1

    # 특징 df 변환
    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index').T
    # 나이 그룹화로 수정
    feature_df['AgeGroup'] =  feature_df['Age'].map(categorize_age)

    # 특징 형태 DTI 일 경우 DTI 형태 특징으로 교체
    if feature_type == 'DTI':
        # 예시 약물 가져와서 피처 초기화
        ti_sum = np.zeros(len(ingredients_df['DTI'][1223]))
        # 각 약물 벡터 합 연산
        for each_drug in pres_list:
            rxcui = rxcui_dict[each_drug]
            each_ti = np.array(ingredients_df['DTI'][rxcui], dtype=np.float64)
            ti_sum += each_ti
        ti_df = pd.DataFrame(ti_sum).T
        # 피처 교체 후 반환
        ti_df = pd.concat([ti_df, feature_df.loc[:, 'Sex_FEMALE':]], axis=1)
        return ti_df
    # 특징 형태 MF 일 경우 MF 형태 특징으로 교체
    elif feature_type == 'MF':
        # 예시 약물 가져와서 피처 초기화
        fp_sum = np.zeros(len(ingredients_df['Fingerprint'][1223]))
        # 각 약물 벡터 합 연산
        for each_drug in pres_list:
            rxcui = rxcui_dict[each_drug]
            each_fp = np.array(ingredients_df['Fingerprint'][rxcui], dtype=np.float64)
            fp_sum += each_fp
        fp_df = pd.DataFrame(fp_sum).T
        # 피처 교체 후 반환
        fp_df = pd.concat([fp_df, feature_df.loc[:, 'Sex_FEMALE':]], axis=1)
        return fp_df
    # 특징 형태 API 그대로 리턴
    else:
        return feature_df