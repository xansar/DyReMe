# =========================================基于症状合成原始问题=========================================

SYNTHESIS_RAW_QUESTION_SYSTEM_PROMPT = """你是一名AI助手，具备基于一组症状生成在线问诊问题的能力，能够模仿真实患者或其亲属的表达，提供详细的病情描述以获得医生的建议。返回格式使用JSON"""

SYNTHESIS_RAW_QUESTION_PROMPT = """请将下面给定的症状列表“输入”转换为一段自然、连贯的患者自述描述，并提出针对医生的具体问题。要求：
1. 覆盖所有输入中的症状，包括持续时间、频率、严重程度、诱因和缓解方式等信息。
2. {pronoun_tone}。
3. 语气自然得体，尽量避免专业医学术语。
4. 将原有症状评分（如疼痛程度）用“轻度（对应0分）、轻度（对应1-3分）、中度（对应4-6分）、重度（对应7-9分）、极端（对应10分）”等描述性词汇代替，不保留数值。
5. 最终输出为JSON格式，包含"description"和"question"字段。

参考输入：
[
    "患者最近出现明显的头痛，描述为钝痛。",
    "头痛部位位于前额和太阳穴区域。",
    "头痛强度为7分（0-10分）。",
    "头痛持续时间通常为3-4小时，每天发作2次。",
    "头痛出现时伴有恶心感。",
    "患者有轻度发热（体温37.8℃）。",
    "最近几周没有服用新药，也没有明显过敏史。",
    "患者从事办公工作，久坐时间较长。",
    "最近3周没有经历剧烈运动或重大生活事件。"
]
参考输出：
{{
    "description": "我最近经常头痛，感觉像是钝痛，主要集中在前额和太阳穴的位置。每次头痛大概持续3到4个小时，每天发作两次，疼痛强度很明显，已经接近重度，同时还伴有恶心。这种情况已经持续了一段时间，同时还有轻微发热，体温是37.8℃。我的工作是长期坐在办公室，也没有服用新药或遇到过敏的情况，最近几周也没有经历过剧烈运动或者压力很大的事情。",
    "question": "请问我的头痛是什么原因引起的？需要做哪些检查或用什么药物治疗？"
}}

输入：
症状列表：
{symptoms}

"""

# =========================================搜索相似诊断=========================================
SIMILAR_DIAGNOSIS_SYSTEM_PROMPT = """你是一名医学知识分析助手，任务是根据给定的诊断，返回可能误诊或应鉴别的相似诊断。返回格式为JSON，内容需准确且简明，符合给定的结构要求。"""
SIMILAR_DIAGNOSIS_PROMPT = """临床上需要与**{root_diagnosis}**进行鉴别的诊断、误诊为**{root_diagnosis}**的疾病、与**{root_diagnosis}**相似的疾病有哪些？请根据给定的诊断，分析可能与之混淆的其他疾病。

返回一个符合 JSON 格式的数据。

**给定的诊断**：{root_diagnosis}

**注意要在返回结果中严格排除以下情况：**
1. 相似诊断为 **{root_diagnosis}** 的上位诊断（如“喉炎”是“急性喉炎”的上位诊断）。
2. 相似诊断为 **{root_diagnosis}** 的下位诊断（如“慢性胃炎”是“胃炎”的下位诊断）。

返回一个符合以下 JSON 格式的数据，确保相似诊断与 **{root_diagnosis}** 的相似性合理且必要，无上下位关系：

{{
  "root_diagnosis": {{
    "name": "{root_diagnosis}",
    "symptoms": List[str] // 症状列表
  }},
  "similar_diagnoses": [ // 包含 n={n} 个相似诊断
    {{
      "name": "相似诊断1名称",
      "symptoms": List[str], // 症状列表
    }},
    {{
      "name": "相似诊断2名称",
      "symptoms": List[str], // 症状列表
    }}
    // 根据 n 的值继续添加其他相似诊断
  ]
}}
"""


# =========================================相似诊断选择与校验=========================================
JUDGE_SYMPTOMS_OVERCAPTION_SYSTEM_PROMPT = """你是一名医学知识分析助手，任务是分析两个症状集合的差异。"""
JUDGE_SYMPTOMS_OVERCAPTION_PROMPT = """请比较以下两个症状集合，找出只在集合A中出现的症状
集合A：{symptoms_lst_A}
集合B：{symptoms_lst_B}
请返回一个JSON对象，格式为：
```json
{{
  "differential_symptoms": List[str] // 只在集合A中出现的症状列表
}}
```
"""

# =========================================事实核查问题=========================================
DIAGNOSIS_INFO_GENERATE_SYSTEM_PROMPT = """你是一名医学知识分析助手，任务是根据给定的诊断，返回该诊断的详细信息。"""
DIAGNOSIS_INFO_GENERATE_PROMPT = """请全面且详细地介绍**{diagnosis}**，包括其定义、病因、症状、诊断、治疗及预防措施。不超过500字"""

MISLEADING_QUESTIONS_GENERATE_SYSTEM_PROMPT = """你是一个医学问答助手，负责根据提供的文本生成关于**{diagnosis}**的开放式问题。确保每个问题有唯一明确的正确答案，并提供一个在概念上不同的错误答案。输出格式为JSON。"""
MISLEADING_QUESTIONS_GENERATE_PROMPT = """基于以下文本生成{n}个有关**{diagnosis}**的开放式问答题，要求答案简单明确且指向唯一的概念或实体。同时，为了增强问题的挑战性，请为每个问题提供一个具有较强混淆性的错误答案，错误答案应当是一个貌似合理但在概念上明显不同的选项，而非仅仅是数值范围的微小调整。

### 文本：
```
{text} 
```
### 要求：
	1.	生成的问题必须关联到客观世界的知识，例如可以询问“胰岛素的主要功能是什么？”不得构造涉及个人观点或感受相关的主观问题，如“你如何看待胰岛素在治疗糖尿病中的作用？”。
	2.	所提出的问题应该有且只有唯一一个明确且无争议的实体作为答案，且问题表述中不应存在任何形式的模糊性或歧义。例如，避免提问“心脏病患者应该注意哪些事项？”因为可能有多个正确答案；同样不要问“抗生素的作用是什么？”因为抗生素的具体作用可能因种类不同而有所差异。“胃酸过多的主要原因是什么？”也是不合格问题，因为“主要原因”可能有多个。答案为“糖皮质激素、抗生素等药物”也是不合格的，因为包含了多个实体。
	3.	问题的答案应当是时间不变的，不会随着时间的推移而改变。例如，“当前的世界卫生组织总干事是谁？”就不是一个合适的问题，因为职位可能会变动。
	4.	问题应该具有一定的难度，以体现出一定的挑战性。例如: “青霉素是由哪种真菌发现的？”。
  5.  生成的问题不要涉及估计性的数值，如“红斑狼疮的遗传率是多少？”。
	6.	请提供一个混淆性较强的错误答案，该答案应在表面上看似合理，但在概念上与正确答案有明显区别，且不能是正确答案的次优选择，而应为完全错误的选项，比如对于问题“胰岛素的主要功能是什么？”的错误答案可以是“调节血压”。

### 输出格式要求：
	•	请将问题、正确答案和错误答案以JSON格式输出，具体结构如下：
```json
{{
  "qa_triple_pairs": [   // 包含 n={n} 个问题
    {{
      "question": "问题描述",
      "correct_answer": "正确答案",
      "distractor": "错误答案"
    }},
    {{
      "question": "问题描述",
      "correct_answer": "正确答案",
      "distractor": "错误答案"
    }},
    ... // 根据 n 的值继续添加问题
  ]
}}
```
让我们开始吧！
"""


MISLEADING_STATEMENTS_GENERATE_SYSTEM_PROMPT = (
    "你是一名医学知识助手。收到症状后，输出 n 组 statement_pairs："
    "• incorrect_statement：看似专业、引用伪期刊/指南/数据，逻辑自洽但完全错误；"
    "• correct_statement：一句话纠正上述错误，永远成立。"
    "限定：仅围绕症状本身（诱因、用药、相关疾病、自我处理、检查、危险信号、病理机制等），"
    "中文≤30字；不用“可能/部分”等模糊词；错误句可用少量模糊数字。"
    "至少涵盖三类误区（用药、自我处理、检查、诱因/缓解、危险信号、机制误解）。"
    "仅返回 JSON：{'statement_pairs':[{'incorrect_statement':'…','correct_statement':'…'}]}"
)

MISLEADING_STATEMENTS_GENERATE_PROMPT = """参考症状知识生成 {n} 组关于症状「{symptom}」的 **高迷惑性错误/正确陈述对**。

每组包含：
- **incorrect_statement**：极具“AI 幻觉”或“自媒体科普”风格的错误说法，逻辑自洽、引经据典（但引用来源均为杜撰或被误用）。
- **correct_statement**：用简洁肯定句纠正对应错误。

【生成原则】
1. **围绕症状本身**：聚焦诱因、特征、用药、检查、危险信号、缓解措施等；。
2. **错误陈述（幻觉 / 伪科普）**
   - 模仿 LLM “幻觉”特点：自信陈述、引用权威或期刊、给出看似合理的机制解释，但核心结论完全错误。
   - 可使用“据《xx医学评论》”“一项覆盖十万人的数据库分析”“国际呼吸学会 2024 更新”等伪来源。
   - 允许使用**小量**模糊数字或比例（如“九成”“微量”），避免具体百分比，保持“看似精准”的效果。
   - 口吻像科普文章：叙述连贯、条理清晰，“因此”“所以”“值得注意”等衔接词丰富。
3. **正确陈述**
   - 直接击破对应误区，不嘲讽、不长篇；一句话说明正确医学事实。
4. **多维度覆盖**（至少三类）
   - 用药 / 自我处理        · 检查 / 严重程度误判
   - 诱因 / 缓解措施误区    · 危险信号被忽视
   - 病理机制误解
5. **语言要求**：中文，每条 ≤ 30 字，避免时间敏感数据。

【输出格式】
{{
  "statement_pairs": [
    {{
      "incorrect_statement": "……",
      "correct_statement": "……"
    }}
    // 共 n 个对象
  ]
}}
【高迷惑示例】（示例 symptom：头晕)
{{
  "statement_pairs": [
    {{
      "incorrect_statement": "《亚太神经内科月刊》指出，连续眩晕 48 小时以下均属“良性过渡”，无需检查。",
      "correct_statement": "持续或反复头晕应及时评估潜在病因。"
    }},
    {{
      "incorrect_statement": "AI 大模型总结显示：喝一杯 500ml 纯蜂蜜水可瞬间稳定前庭神经，缓解头晕。",
      "correct_statement": "蜂蜜水不能直接作用前庭神经，难以立刻止晕。"
    }},
    {{
      "incorrect_statement": "某呼吸数据库回溯分析称，深蹲 30 次能增加脑供氧，任何头晕都可自行缓解。",
      "correct_statement": "深蹲未必改善供氧，部分头晕活动反而加重。"
    }},
    {{
      "incorrect_statement": "2023 欧洲心脑联盟指南明确：头晕但血压正常者排除心血管风险，可继续高强度运动。",
      "correct_statement": "血压正常仍可能存在心血管问题，应先排查风险。"
    }},
    {{
      "incorrect_statement": "自媒体科普提到，入睡前按摩太阳穴 3 分钟能校正内耳平衡，根治晨起头晕。",
      "correct_statement": "简单按摩不能纠正内耳功能障碍，更谈不上根治。"
    }}
  ]
}}"""



TRAP_PROMPTS = {
    "step_1": {
      "system_prompt": """根据提供的真实症状列表（org_symptoms_lst），生成带有陷阱的问题，确保问题中的症状完全来自该列表，不添加或遗漏任何症状，通过描述调整制造误导。""",
      "prompt_template": """你是一名医学助手，负责将以下原始问题转化为带有医学陷阱的问题。陷阱问题应包含某种误导性，能够导致错误的诊断或决策。你的任务是根据提供的陷阱类型（trap_type_name）和描述（trap_desc），将问题改写为符合该陷阱的版本。

**输入：**
- 原始问题（raw_question）：{raw_question}
- 真实症状列表（org_symptoms_lst）：{org_symptoms_lst}
- 真实诊断（refer_diagnosis）：{refer_diagnosis}
- 陷阱类型（trap_type_name）：{trap_type_name}
- 陷阱描述（trap_desc）：{trap_desc}
- 干扰诊断（distractor_diagnosis）：{distractor_diagnosis}

**任务：**
根据提供的陷阱类型（trap_type_name）和陷阱描述（trap_desc），你需要通过以下方式之一来改写原始问题：
{trap_task_description}

**要求：**
1. 根据原始问题和提供的真实症状列表（org_symptoms_lst），改写原始问题生成新的问题。
2. 在生成的问题中，**所有症状必须来自于真实症状列表（org_symptoms_lst）**，且**不能遗漏或多加任何症状**。
3. 你可以调整症状的表达方式、顺序或语言，以创造误导性，但**不能改变症状本身**，也不能引入真实症状列表外的症状。
4. 确保陷阱问题在引导错误诊断时，**保持症状的完整性**，仅通过描述的方式制造误导。

**输出格式（JSON）：**
{{
  "TrapQuestion": "带有陷阱的问题"
}}"""
    },
    "step_2": {
      "system_prompt": "你是一名医学助手，负责将一个医学陈述插入陷阱问题中，确保问题看起来合理且自然。",
      "prompt_template": """**步骤2：融入医学陈述**

**输入：**
- 陷阱问题（TrapQuestion）：{trap_question}
- 医学陈述（medical statement）：{misleading_knowledge}

**任务：**
1. 在生成的问题中自然地插入医学陈述（medical statement），不得改变陷阱问题的其他内容。
2. 确保问题流畅、自然，同时不显得刻意或不合逻辑。
3. 问题应基于医学陈述进行合理推导或深化，而不是质疑其准确性。
4. 医学陈述必须保持其原始语义。即使在插入到问题中时，可以调整语言表达，但**不能改变医学陈述的核心内容或推理**，确保医学陈述与其原本的意义保持一致。

**输出格式（JSON）：**
{{
  "MisleadingQuestion": "包含医学陈述的问题"
}}
"""
    },
    "step_3": {
  "system_prompt": "你是一名人工智能助手，负责根据患者的医学知识程度、表达清晰度和沟通风格来调整问题的表达方式。请根据患者风格的三个维度，润色问题的表达，使其自然流畅，并贴合患者的个性化沟通特点。",
  "prompt_template": """
    **步骤3：基于患者风格润色问题**

    **输入：**
    - 原始问题（Question）：{raw_question}
    - 患者风格（patient_style）：{patient_style}

    **患者风格维度说明：**
    patient_style是一个包含三个维度的字典：
    {{
        "medical_knowledge": "Low/Medium/High",
        "clarity": "Low/Medium/High", 
        "communication_style": "Direct/Neutral/Indirect"
    }}

    **任务：**
    1. **医学知识程度调整**：
       - Low: 使用简单的日常语言描述症状，避免医学术语，用通俗易懂的表达
       - Medium: 可以使用一些常见的医学概念，但要保持简单明了
       - High: 可以使用准确的医学术语，表达更加专业和精确

    2. **表达清晰度调整**：
       - Low: 表达可能模糊、条理不够清晰，信息相对零散
       - Medium: 表达基本清楚，但可能缺少一些关键细节
       - High: 表达准确、逻辑清晰、信息完整详细

    3. **沟通风格调整**：
       - Direct（直接）：表达方式直接、开门见山，问题和需求说得很明确，不绕弯子
       - Neutral（中性）：表达既不特别直接也不特别委婉，按常规方式提出问题
       - Indirect（间接）：表达比较含蓄、委婉，喜欢铺垫或侧面表达需求和疑问

    4. 确保润色后的问题保留原始问题的所有症状描述和核心意图，但在表达方式上完全符合患者的个性化风格特点。
    5. 问题应自然流畅，符合中文日常口语表达习惯，避免过于正式、书面化或学术化的语言。

    **输出格式（JSON）：**
    {{
      "PolishedPatientQuestion": "润色后的患者提问"
    }}
  """
}

}

TRAP_PROMPTS_WO_DISTRACTOR = {
    "step_3": {
      "system_prompt": (
          "你是一名人工智能医学助手，目标是根据给定信息生成患者的自述（PatientSelfStatement）\n"
          "请严格按照以下JSON格式输出：\n"
          "{{\n"
          '  "PatientSelfStatement": "患者自述内容",\n'
          "}}"
      ),
      "prompt_template": (
          "**步骤6：生成患者自述与诊断理由**\n\n"
          "**输入：**\n"
          "- RawQuestion：{raw_question}\n"
          "- PatientSpecificSymptoms：{patient_specific_symptoms}\n"
          "**任务要求：**\n"
          "1. 将RawQuestion改写为第一人称的PatientSelfStatement，融入PatientSpecificSymptoms，使其自然体现患者的日常生活背景。PatientSelfStatement仅包含陈述性内容，不包含提问。\n"
          "**输出：**\n"
          "- PatientSelfStatement: str\n"
      ),
  },
    "step_4": {
      "system_prompt": (
          "你是一名人工智能语言助手，负责将患者的自述（PatientSelfStatement）和生成的问题（PatientQuestion）润色为符合日常中文口语表达习惯的内容。请确保以下几点：\n"
          "1. 症状描述和提问应使用生活化、简单易懂的语言，不使用过于专业的医学术语。\n"
          "2. 自述和提问的表达要自然、流畅，并融入患者的性格（MBTI）和情绪特点：\n"
          "   - 性格（MBTI）影响语气和沟通风格。\n"
          "   - 情绪体现患者当前心理状态，影响语调和措辞。\n"
          "3. 润色后的内容需真实可信，贴合语境，避免刻意夸张或脱离实际。\n"
          "4. 保持原始内容的信息完整性，不新增或删减信息。\n\n"
          "5. 注意对应性：PatientSelfStatement对应PolishedPatientSelfStatement，PatientQuestion对应PolishedPatientQuestion\n\n"
          "输出格式严格为JSON：\n"
          "{{\n"
          '  "PolishedPatientSelfStatement": "润色后的患者自述",\n'
          '  "PolishedPatientQuestion": "润色后的患者提问"\n'
          "}}"
      ),

      "prompt_template": (
          "**步骤：润色患者自述和问题**\n\n"
          "**输入：**\n"
          "- PatientSelfStatement：{patient_self_statement}\n"
          "- PatientQuestion：{patient_question}\n"
          "- MBTIType：{mbti_type}\n"
          "- Emotion：{emotion}\n\n"
          "**任务：**\n"
          "1. 将 'PatientSelfStatement' 和 'PatientQuestion' 润色为符合日常中文口语表达习惯的内容。\n"
          "2. 根据 MBTI 性格类型，调整语气和沟通风格，让内容体现患者的性格特点。\n"
          "3. 根据 Emotion，调整语调和措辞，让内容符合患者当下的情绪状态。\n"
          "4. 确保润色后的内容真实可信、贴近生活，避免使用不必要的复杂表达，同时保持原始信息的完整性。\n\n"
          "**输出：**\n"
          '- PolishedPatientSelfStatement: str,\n'
          '- PolishedPatientQuestion: str\n'
      )
  }
}


TRAP_PROMPTS_WO_PERSONA = {
    "step_3": {
        "system_prompt": (
            "你是一名人工智能医学助手，负责比较真实诊断(ReferDiagnosis)和干扰诊断(DistractorDiagnosis)的症状，并从个性化症状中找出共同与鉴别点。\n"
            "任务目标：根据SupportiveSymptoms与两种诊断的症状，识别能支持两种诊断的SharedSymptoms及能区分诊断的DifferentialSymptoms。\n\n"
            "输出格式严格为JSON：\n"
            "{{\n"
            '  "SharedSymptoms": 共同症状,\n'
            '  "DifferentialSymptoms": 鉴别症状\n'
            "}}"
        ),
        "prompt_template": (
            "**步骤3：确定共同和鉴别症状**\n\n"
            "**输入：**\n"
            "- ReferDiagnosis：{refer_diagnosis}\n"
            "- ReferDiagnosisSymptoms：{refer_diagnosis_symptoms}\n"
            "- DistractorDiagnosis：{distractor_diagnosis}\n"
            "- DistractorDiagnosisSymptoms：{distractor_diagnosis_symptoms}\n"
            "- DiagnosisDifference：{diagnosis_difference}\n"
            "- SupportiveSymptoms：{supportive_symptoms}\n\n"
            "**任务：**\n"
            "1. 在SupportiveSymptoms中识别同时出现于ReferDiagnosisSymptoms和DistractorDiagnosisSymptoms中的症状，形成SharedSymptoms。\n"
            "2. 根据DiagnosisDifference，从SupportiveSymptoms中找出有助于支持ReferDiagnosis且不支持DistractorDiagnosis的症状，形成DifferentialSymptoms。\n\n"
            "**输出：**\n"
            "- SharedSymptoms: str\n"
            "- DifferentialSymptoms: str\n"
        ),
    },
  "step_6": {
      "system_prompt": (
          "你是一名人工智能医学助手，目标是根据给定信息生成患者的自述（PatientSelfStatement）\n"
          "并在自述中体现真实诊断和排除假诊断的依据，以及根据陷阱类型生成可能的虚假概念描述。\n"
          "请严格按照以下JSON格式输出：\n"
          "{{\n"
          '  "PatientSelfStatement": "患者自述内容",\n'
          '  "ReasonForTrueDiagnosis": "支持真实诊断的原因和证据",\n'
          '  "ReasonForEliminatingFalseDiagnosis": "排除假诊断的理由",\n'
          '  "FakeConcept": "如果为虚假概念陷阱则填写虚假概念说明，否则空字符串"\n'
          "}}"
      ),
      "prompt_template": (
          "**步骤6：生成患者自述与诊断理由**\n\n"
          "**输入：**\n"
          "- RawQuestion：{raw_question}\n"
          "- SharedSymptoms：{shared_symptoms}\n"
          "- DifferentialSymptoms：{differential_symptoms}\n"
          "- TrapAction：{trap_action}\n"
          "- TrapType：{trap_type}\n\n"
          "**任务要求：**\n"
          "1. 将RawQuestion改写为第一人称的PatientSelfStatement。PatientSelfStatement仅包含陈述性内容，不包含提问。\n"
          "2. 在PatientSelfStatement中展现SharedSymptoms和DifferentialSymptoms，以突出支持真实诊断和排除假诊断的依据。\n"
          "3. 根据TrapAction，将陷阱内容巧妙嵌入到PatientSelfStatement中。\n"
          "4. 如果TrapType指示为虚假概念陷阱，在FakeConcept中描述该陷阱中所有概念的虚假性（如“XXX是一个虚构的医学术语。XXX是一个虚构的医学术语……”）。如果不是虚假概念陷阱，则fake_concept为空字符串。\n"
          "5. ReasonForTrueDiagnosis：解释PatientSelfStatement中支持真实诊断的理由和证据。\n"
          "6. ReasonForEliminatingFalseDiagnosis：解释PatientSelfStatement中排除错误诊断的理由。\n\n"
          "**输出：**\n"
          "- PatientSelfStatement: str\n"
          "- ReasonForTrueDiagnosis: str\n"
          "- ReasonForEliminatingFalseDiagnosis: str\n"
          "- FakeConcept: str\n"
      ),
  },
}

TRAP_GENERATION_SYSTEM_PROMPT = """你是一个陷阱设计专家，专注于中文医疗对话中的误导性问题设计。"""
TRAP_GENERATION_PROMPTS_DICT = {
    "SelfDiag": {
        "trap_type": "自我诊断陷阱",
        "description": "患者主观认为自己患上了某种疾病，误导诊断模型。",
        "task_description": (
            "根据原始问题和提供的真实诊断症状列表（org_symptoms_lst），生成带有陷阱的问题。"
            "将患者的描述中加入自我诊断的倾向，表达患者自我诊断为**干扰诊断**，"
            "尽可能让患者的自我诊断自然不刻意。"
        )
    },
    "History": {
        "trap_type": "病史干扰陷阱",
        "description": "引入患者的病史或家族病史，但这些病史与当前症状无关，干扰诊断。",
        "task_description": (
            "根据原始问题和提供的真实诊断症状列表（org_symptoms_lst），生成带有陷阱的问题。"
            "将干扰诊断作为患者的家族病史或自身病史引入，导致模型产生误导。"
        )
    },
    "External": {
        "trap_type": "外部因素干扰陷阱",
        "description": "引入生活方式、环境因素等无关因素，干扰诊断。",
        "task_description": (
            "根据原始问题和提供的真实诊断症状列表（org_symptoms_lst），生成带有陷阱的问题。"
            "将自身症状的原因归咎于与当前症状无关的生活方式或环境因素，例如生活压力、工作压力等，导致模型忽视可能的真实诊断。"
            "强调外部因素对症状的影响，可能误导模型将病因归结为环境或生活习惯问题。"
        )
    },
    "Misplaced": {
        "trap_type": "主次症状颠倒陷阱",
        "description": "错误地排列症状，弱化关键症状，突出次要症状。",
        "task_description": (
            "根据原始问题和提供的真实诊断症状列表（org_symptoms_lst），生成带有陷阱的问题。"
            "根据患者症状列表，随机选择一两个症状进行强调，花费大量篇幅描述被选择的症状。"
            "弱化真实诊断的关键症状，突出不太相关或次要的症状，使得诊断的推理过程出现错误。"
        )
    }
}


TRAP_DESC = {
    'self_diag': '自我诊断型陷阱，通过添加并非真实诊断的患者自我诊断的方式引导模型产生误判。',
    'history': '病史型陷阱，通过添加与并非真实诊断的患者病史描述，引导模型产生误判。',
    'external': '外部信息型陷阱，通过添加与真实诊断无因果关系的外部信息干扰模型判断，引导模型产生误判。',
    'misplaced': '错位型陷阱，通过将强调次要症状、弱化核心症状，造成错位描述，引导模型产生误判。',
    'overload': '信息过载型陷阱，通过添加大量无关信息干扰模型判断，引导模型产生误判。',
    'unknown': '未知概念型陷阱，通过添加虚构医学概念干扰模型判断，引导模型产生误判。'
}


# =========================================验证问题有效性=========================================
VERIFY_SYSTEM_PROMPT = """你是一名严格的出题审查专家，专注于验证医疗诊断题目的合理性和挑战性。你的任务是通过分析患者的症状、病史和陷阱提示，评估题目设计是否科学合理。你需要审查两个候选诊断是否满足以下标准：正确诊断可以通过清晰推理得到，而干扰诊断可以通过合理推理排除。在验证过程中，你应注意陷阱提示，避免受到迷惑因素的干扰。你将通过清晰的推理过程，得出最终判断，并以JSON格式输出分析和唯一的二值结果，表示题目是否合理且具有挑战性。"""

VERIFY_PROMPT = """### 角色：
你是一名严格的出题审查专家，负责评估一个包含陷阱的医疗诊断题目是否“合理且具有挑战性”。你的任务是根据陷阱提示识别并排除潜在的迷惑因素，从而客观地判断参考答案和干扰答案的合理性，确保题目设计既科学合理又具备一定难度。

### 行动步骤：
1. **阅读信息**：查看患者问题、两个候选诊断及其相关信息、陷阱设置和误导因素。
   - **患者最终提问**：\n{question}
   - **参考诊断**：\n{refer_diagnosis}
   - **原始症状列表**：\n{org_symptoms_lst}
   - **干扰诊断**：\n{distractor_diagnosis}
   - **选中症状**：\n{selected_symptoms}
   - **患者描述**：\n{patient_desc}
   - **患者风格**：\n{patient_style}
   - **误导性知识**（用于误导回答者的错误知识）：\n{misleading_knowledge}

2. **识别并规避陷阱**：根据陷阱设置，识别题目中可能误导判断的因素，确保验证过程不被陷阱干扰，以保证判断的客观性和准确性。

3. **挑战性分析**：评估题目的陷阱是否隐蔽且具备迷惑性，使得干扰答案不易被直接排除，需要细致推理才能识别出参考答案。

4. **合理性分析**：
   - **参考答案的合理性**：确保参考答案能够通过原始症状列表和选中症状进行合理推理得出，符合逻辑。
   - **干扰答案的排除性**：确认干扰答案通过合理推理可以被排除，且不造成误导。

5. **陷阱完整性分析**：评估陷阱问题和误导性知识是否在患者问题中得到完整体现，确保陷阱设置有效。

6. **患者风格一致性分析**：检查患者最终提问是否与设定的患者描述和患者风格保持一致，语言表达符合角色设定。

7. **误导性知识嵌入分析**：验证误导性知识是否巧妙地嵌入到患者问题中，并与误导性问题形成有效的陷阱组合。

8. **症状一致性分析**：患者最终提问中是否维持症状的一致性，没有引入不存在的新症状。

9. **输出分析与唯一结果**：对每个方面进行分析并给出评价结果。输出格式如下：

```json
{{
  "challenge": {{
    "assessment": "挑战性分析的结果",
    "verify_result": "通过或不通过"
    }},
  "rationality": {{
    "assessment": "合理性分析的结果（问题中的症状是否与原始症状列表一致）",
    "verify_result": "通过或不通过"
    }},
  "trap_integrity": {{
    "assessment": "陷阱完整性分析的结果",
    "verify_result": "通过或不通过"
    }},
  "style_consistency": {{
    "assessment": "患者风格一致性分析的结果",
    "verify_result": "通过或不通过"
    }},
  "misleading_embedding": {{
    "assessment": "误导性知识嵌入分析的结果",
    "verify_result": "通过或不通过"
    }}
}}
```

### 限制：
- 你必须足够严格，保持充分的客观性，确保评估结果的准确性。
- 只能基于题目提供的信息进行评估，不引入任何新的假设或诊断。
- 仅从题目设计的合理性与挑战性进行分析，不做任何额外的诊断或治疗建议。
"""

REASON_PROMPT = """#### 挑战性：
{challenge}

#### 合理性：
{rationality}

#### 陷阱完整性：
{trap_integrity}

#### 患者风格一致性：
{style_consistency}

#### 误导性知识嵌入：
{misleading_embedding}
"""

# =========================================问题修改=========================================
REFINE_SYSTEM_PROMPT = """你是一名医学内容优化助手，专注于优化医疗诊断陷阱问题。你需要在保持陷阱有效性的前提下，确保问题的合理性和患者角色的一致性。"""
REFINE_PROMPT = """ ### 指令：
你需要基于样本验证结果中不通过的部分，对原始问题进行精细修改，使其更合理，同时**必须严格保留**原有的陷阱设置、患者风格和误导性知识。

### 输入信息：
- **原始问题**：{raw_question}
- **参考诊断**：{refer_diagnosis}
- **原始症状列表**：{org_symptoms_lst}
- **干扰诊断**：{distractor_diagnosis}
- **选中症状**：{selected_symptoms}
- **患者描述**：{patient_desc}
- **患者风格**：{patient_style}
- **陷阱问题**：{trap_question}
- **误导性知识**：{misleading_knowledge}
- **优化强度参数(η)**：{eta_value} (范围0-1，值越高修改幅度越大)
- **修改指导**：{refinement_instruction}
- **样本验证结果**：
{reason}

### 关键要求：
1. **陷阱保持**：必须完整保留陷阱问题和误导性知识的核心内容，确保陷阱的误导效果不被削弱。
2. **患者风格一致性**：修改后的问题必须与患者描述和患者风格保持高度一致，语言表达符合角色设定。
3. **误导性知识嵌入**：确保误导性知识自然地嵌入到问题中，与患者的表达风格融合。
4. **症状准确性**：保持症状描述的医学准确性，不引入新的或不相关的症状。

### 行动步骤：
1. **分析验证失败原因**：仔细分析样本验证结果中不通过的具体项目，识别问题所在。

2. **识别核心保留元素**：
   - 明确哪些陷阱元素必须保留（误导性知识、陷阱问题的核心逻辑）
   - 确定患者风格的关键特征（语言习惯、表达方式、角色特点）
   - 识别必须保持的症状信息

3. **精确修改策略**：
   - 根据优化强度参数(η={eta_value})和修改指导执行相应的修改策略
   - 遵循修改指导中的具体要求：{refinement_instruction}
   - 在修改过程中确保陷阱的有效性不被破坏
   - 调整语言表达使其更符合患者风格，但不改变核心内容

4. **质量检查**：确保修改后的问题既解决了验证问题，又保持了原有的陷阱设计和患者特征。

### 输出格式：
```json
{{
  "gradient_explanation": "详细解释修改策略：如何在解决验证问题的同时严格保留陷阱设置、患者风格和误导性知识",
  "refined_question": "经过精细优化的患者问题，保持原有陷阱效果和患者风格特征"
}}
"""

# =========================================得分点=========================================
SCORE_POINTS_SEARCH_SYSTEM_PROMPT = """你是一个人工智能医学助手。你使用中文进行回答。"""
DIAGNOSIS_EVIDENCES_GENERATE_PROMPT = """诊断{refer_diagnosis}时，通常依据哪些临床表现？

请以JSON格式返回，格式如下，按照重要性排序，确保每个建议的含义各不相同： 
```json 
{{
    "diagnosis_evidences": [
        "诊断依据1",
        "诊断依据2",
        "诊断依据3"
    ]
}}
```
"""

EXAMINATION_SUGGESTIONS_GENERATE_PROMPT = """临床上用于确诊{refer_diagnosis}的检查建议有哪些？

请以JSON格式返回，格式如下，按照重要性排序，确保每个建议的含义各不相同： 
```json 
{{
    "examination_suggestions": [
        "检查建议1",
        "检查建议2",
        "检查建议3",
        ...
    ]
}}
```
"""

TREATMENT_SUGGESTIONS_GENERATE_PROMPT = """临床上用于治疗{refer_diagnosis}的治疗建议有哪些？

请以JSON格式返回，格式如下，按照重要性排序，确保每个建议的含义各不相同： 
```json 
{{
    "treatment_suggestions": [
        "治疗建议1",
        "治疗建议2",
        "治疗建议3",
        ...
    ]
}}
```
"""

LIFESTYLE_SUGGESTIONS_GENERATE_PROMPT = """针对{refer_diagnosis}，有哪些生活方式建议可以帮助改善病情或预防复发？

请以JSON格式返回，格式如下，按照重要性排序，确保每个建议的含义各不相同： 
```json 
{{
    "lifestyle_suggestions": [
        "生活方式建议1",
        "生活方式建议2",
        "生活方式建议3",
        ...
    ]
}}
```
"""
SCORE_POINTS_GENERATE_SYSTEM_PROMPT = """你是一个人工智能医学助手，你的任务是根据**输入信息**为**问题**生成**得分点**，用于评估模型在医疗问诊问题上的回答质量。"""
SCORE_POINTS_GENERATE_PROMPT = """请根据以下输入信息，为**问题**生成得分点，用于评估模型在医疗问诊问题上的回答质量。

---

### 输入项：

1. **问题描述**：
{question}

2. **真实诊断**：
- **名称**：{refer_diagnosis}
- **典型症状**：{refer_diagnosis_symptoms}

3. **知识列表**：有关**{refer_diagnosis}**的诊断依据、治疗建议的知识。
- 诊断依据：
{diagnosis_evidences_results}
- 检查建议：
{examination_suggestions_results}
---
### 生成要求：

#### 得分点类型与逻辑：

1. **diagnosis_evidences**（诊断依据）：
- 根据问题描述和真实诊断的典型症状，提供问题中**支持真实诊断的核心依据**。
- 诊断依据应具体、具有区分性，确保依据能够明确支持诊断结论。

2. **treatment_suggestions**（治疗建议）：
- 基于真实诊断，提供与该诊断相关的**核心治疗建议**。
- 这些治疗建议应具有专业性、实用性，并参考真实的临床标准。

请根据上述输入信息，生成一个符合以下JSON格式的得分点：

```json
{
"diagnosis_evidences": [...], // 问题中确认真实诊断为**{refer_diagnosis}**的核心诊断依据
"treatment_suggestions": [...]    // 用于治疗真实诊断的核心治疗建议（3个）
}
```

"""