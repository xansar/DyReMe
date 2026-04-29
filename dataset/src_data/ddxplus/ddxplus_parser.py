import json
import csv
import requests
import time

from tinydb import TinyDB, Query
import random
random.seed(42)

# 初始化 TinyDB
DB_PATH = 'dataset/ddxplus/raw_patients.json'
db = TinyDB(DB_PATH, sort_keys=True, indent=4, separators=(',', ': '))
patients_table = db.table('patients')
# 清除表内容
patients_table.truncate()
print(patients_table.all())

# Directory where the dataset is stored
DATA_DIRECTORY = 'dataset/src_data/ddxplus/'
MAX_DECODE = 300

def load_json(file_path):
    with open(DATA_DIRECTORY + file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def decode_evidence(evidence_code, evidences_data):
    parts = evidence_code.split('_@_')
    base_evidence_code = parts[0]
    evidence_value = parts[1] if len(parts) > 1 else None
    evidence = evidences_data.get(base_evidence_code, {})

    if evidence['data_type'] == 'B':
        return {"question": evidence['question_en'], "value": "Yes" if evidence_value is None else "No"}
    elif evidence['data_type'] in ['C', 'M']:
        # If evidence_value is a digit, use it directly as the value
        if evidence_value is not None and evidence_value.isdigit():
            return {"question": evidence['question_en'], "value": evidence_value}
        else:
            value_meaning = evidence['value_meaning'].get(evidence_value, {})
            return {"question": evidence['question_en'], "value": value_meaning.get('en', '')}


def decode_evidence_value(evidence_code, evidence_data):
    if evidence_data['data_type'] == 'N':
        return str(evidence_code)
    elif evidence_data['data_type'] in ['C', 'M']:
        value_meaning = evidence_data['value_meaning'].get(evidence_code, {})
        return value_meaning.get('en', '')
    else:
        return "Yes" if evidence_code else "No"

def decode_pathology(pathology_code, conditions_data, evidences_data):
    pathology = conditions_data.get(pathology_code, {})
    decoded_symptoms = []
    decoded_antecedents = []

    for symptom in pathology.get('symptoms', []):
        evidence = evidences_data.get(symptom, {})
        symptom_name = evidence.get('question_en', 'Unknown Symptom')
        decoded_symptoms.append(symptom_name)

    for antecedent in pathology.get('antecedents', []):
        evidence = evidences_data.get(antecedent, {})
        antecedent_name = evidence.get('question_en', 'Unknown Antecedent')
        decoded_antecedents.append(antecedent_name)

    return {
        "name": pathology.get('cond-name-eng', ''),
        'icd10-id': pathology.get('icd10-id', ''),
        "symptoms": decoded_symptoms,
        "antecedents": decoded_antecedents,
        "severity": pathology.get('severity', '')
    }

def process_patient(patient, evidences_data, conditions_data):
    decoded_pathology = decode_pathology(patient["PATHOLOGY"], conditions_data, evidences_data)
    decoded_patient = {
        "sex": patient["SEX"],
        "age": patient["AGE"],
        "diagnosis": decoded_pathology["name"],
        "icd10-id": decoded_pathology["icd10-id"],
        "pathology": decoded_pathology,
        "condition": [],
        "initial_condition": decode_evidence(patient["INITIAL_EVIDENCE"], evidences_data),
        "differential_diagnosis": {}
    }

    for evidence in patient["EVIDENCES"]:
        decoded_patient["condition"].append(decode_evidence(evidence, evidences_data))

    for diagnosis in eval(patient["DIFFERENTIAL_DIAGNOSIS"]):
        decoded_patient["differential_diagnosis"][decode_pathology(diagnosis[0], conditions_data, evidences_data)["name"]] = diagnosis[1]

    return decoded_patient

def count_total_rows(file_path):
    total_rows = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)  # Skip the header row
        for _ in csv_reader:
            total_rows += 1
    return total_rows


def main(max_decode):
    evidences_data = load_json('release_evidences.json')
    conditions_data = load_json('release_conditions.json')
    patient_data_file = DATA_DIRECTORY + 'test.csv'  # 数据集文件路径

    # 读取所有病人数据到内存中
    with open(patient_data_file, 'r', encoding='utf-8') as file:
        csv_reader = list(csv.DictReader(file))

    # 随机抽取指定数量的病人
    random_patients = random.sample(csv_reader, k=max_decode)
    print(f"Processing {len(random_patients)} patients")

    # 处理随机抽取的病人
    QueryModel = Query()
    for idx, row in enumerate(random_patients):
        patient_id = 'patient_' + str(idx)
        if patients_table.contains(QueryModel.patient_id == patient_id):
            continue
        # time.sleep(0.1)  # 限制请求频率，避免过快
        row["EVIDENCES"] = row["EVIDENCES"].strip("[]").replace("'", "").split(", ")
        # print(f"\n\n{row}\n\n")
        patient = process_patient(row, evidences_data, conditions_data)
        patient['patient_id'] = 'patient_' + str(idx)

        # 保存到 TinyDB
        patients_table.insert(patient)

    print(f"{len(random_patients)} patients processed and saved to TinyDB")



if __name__ == "__main__":
    main(MAX_DECODE)
