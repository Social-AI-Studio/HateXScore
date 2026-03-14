from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import jieba
import numpy as np
import pandas as pd
import spacy
from konlpy.tag import Okt

from .cc import consistency_check
from .htc import extract_answer_yes, htc
from .qf import overlap_quoted_phrases, quotation_faithfulness, quoted_phrases
from .tgi import lemmatize, tgi


class ReasoningMetricsEvaluator:
    def __init__(
        self,
        language: str,
        spacy_model: str = "en_core_web_sm",
        metric_weights: Dict[str, float] | None = None,
        runtime_args=None,
    ):
        self.language = language
        self.runtime_args = runtime_args
        self.metric_weights = metric_weights or {
            "HTC": 1.0,
            "Quotation Faithfulness": 1.0,
            "TGI": 1.0,
            "Consistency Check": 1.0,
        }

        if language == "zh":
            self.nlp = None
        elif language == "kr":
            self.nlp = None
            self.okt = Okt()
        else:
            self.nlp = spacy.load(spacy_model)

    def _htc(self, reasoning: str, pred_label: str) -> float:
        return htc(reasoning, pred_label)

    def _mask_rationales(self, text: str, quotes: List[str]) -> str:
        from .qf import mask_rationales

        return mask_rationales(text, quotes)

    def _norm(self, t: str) -> List[str]:
        from .qf import norm

        return norm(self.language, t, getattr(self, "okt", None))

    def _quoted_phrases(self, text: str, reasoning: str) -> List[str]:
        return quoted_phrases(self.language, text, reasoning, getattr(self, "okt", None))

    def _overlap_quoted_phrases(self, text: str, analysis: str) -> List[str]:
        return overlap_quoted_phrases(self.language, text, analysis, getattr(self, "okt", None))

    def extract_answer_number(self, sentence: str) -> float:
        from .qf import extract_answer_number

        return extract_answer_number(sentence)

    def _predict_proba(self, text: str) -> float:
        from .qf import predict_proba

        return predict_proba(text, self.runtime_args)

    def _quotation_faithfulness(self, text: str, reasoning: str, prediction: str) -> float:
        return quotation_faithfulness(
            text=text,
            reasoning=reasoning,
            prediction=prediction,
            language=self.language,
            args=self.runtime_args,
            okt=getattr(self, "okt", None),
        )

    def _lemmatize(self, text: str) -> List[str]:
        return lemmatize(self.language, text, nlp=self.nlp, okt=getattr(self, "okt", None))

    def _tgi(self, reasoning: str, quotes: List[str], target_group_list: List[str]) -> int:
        return tgi(
            reasoning=reasoning,
            quotes=quotes,
            target_group_list=target_group_list,
            language=self.language,
            nlp=self.nlp,
            okt=getattr(self, "okt", None),
        )

    def _consistency_check(self, pred_label: str, qf: float, tgi_score: int) -> int:
        return consistency_check(pred_label, qf, tgi_score)

    def _compute_hatexscore(self, scores: Dict[str, float]) -> float:
        total_weight = sum(self.metric_weights.values())
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(scores[key] * self.metric_weights.get(key, 0.0) for key in scores.keys())
        return weighted_sum / total_weight

    def evaluate_sample(self, sample: Dict, target_group_list: List[str]) -> Dict:
        text = sample["text"]
        reasoning = sample["reasoning"]
        gold_label = sample["gold_label"].lower()
        predicted_label = sample["prediction"].lower()

        htc_score = self._htc(reasoning, predicted_label)
        qf_score = self._quotation_faithfulness(text, reasoning, predicted_label)
        quotes = self._quoted_phrases(text, reasoning)
        tgi_score = self._tgi(reasoning, quotes, target_group_list)
        cc_score = self._consistency_check(predicted_label, qf_score, tgi_score)
        print("find nooooooooone", htc_score, qf_score, tgi_score, cc_score)

        metric_scores = {
            "HTC": htc_score,
            "Quotation Faithfulness": qf_score,
            "TGI": tgi_score,
            "Consistency Check": cc_score,
        }
        hatexscore = self._compute_hatexscore(metric_scores)

        return {
            "HTC": htc_score,
            "Quotation Faithfulness": qf_score,
            "TGI": tgi_score,
            "Consistency Check": cc_score,
            "HateXScore": round(hatexscore, 3),
        }

    def evaluate_dataset(self, data: List[Dict], target_group_list: List[str]) -> Tuple[List[Dict], Dict[str, float]]:
        records = [self.evaluate_sample(sample, target_group_list) for sample in data]
        aggregate = {k: float(np.mean([r[k] for r in records])) for k in records[0].keys()}
        return records, aggregate


def write_json(data, path):
    f = open(path, mode="a", encoding="utf-8")
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
    f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='hatexplain')
    parser.add_argument("--data_path", default="hatexplain_gpt.json")
    parser.add_argument("--input_csv", default="hatexplain.csv")
    parser.add_argument("--output_dir", default="...")
    parser.add_argument("--model", default='gpt')
    parser.add_argument("--protected_group", default='un_en')
    parser.add_argument("--lang", default='en')
    parser.add_argument("--weight_htc", type=float, default=1.0)
    parser.add_argument("--weight_qf", type=float, default=1.0)
    parser.add_argument("--weight_tgi", type=float, default=1.0)
    parser.add_argument("--weight_cc", type=float, default=1.0)
    return parser.parse_args()


def get_protected_classes(protected_group: str):
    if protected_group == "facebook":
        return {
            "race": ["black", "white", "asian", "latino", "jewish", "arab"],
            "religion": ["muslim", "christian", "jew"],
            "gender": ["women", "men", "transgender"],
            "sexual orientation": ["gay", "lesbian", "bisexual", "queer"],
        }
    elif protected_group == "youtube":
        return {
            "race": ["african american", "asian", "hispanic", "native american"],
            "religion": ["muslim", "christian"],
            "gender": ["women", "men"],
            "sexual orientation": ["gay", "lesbian"],
        }
    elif protected_group == "un_en":
        return {
            "race": [
                "african", "african american", "afro-caribbean", "arab", "armenian", "asian", "assyrian",
                "australian aboriginal", "balkan", "bangladeshi", "basque", "berber", "black", "brazilian",
                "british", "bulgarian", "burmese", "caucasian", "cambodian", "cajun", "chinese", "cuban",
                "czech", "danish", "dominican", "dutch", "egyptian", "english", "estonian", "filipino",
                "finnish", "french", "georgian", "german", "greek", "gypsy / roma", "haitian", "han chinese",
                "hispanic", "hungarian", "icelandic", "indian", "indigenous", "inuit", "iranian", "iraqi",
                "irish", "israeli", "italian", "jamaican", "japanese", "jewish", "jordanian", "korean",
                "kurdish", "laotian", "latino", "lebanese", "malay", "maori", "mexican", "middle eastern",
                "mongolian", "moroccan", "muslim", "native american", "new zealander", "nigerian", "nordic",
                "norwegian", "pakistani", "palestinian", "persian", "polish", "portuguese", "puerto rican",
                "romanian", "russian", "saudi arabian", "scottish", "serbian", "singaporean", "somali",
                "south african", "spanish", "sri lankan", "sudanese", "swedish", "swiss", "syrian", "taiwanese",
                "thai", "tibetan", "turkish", "ukrainian", "vietnamese", "welsh", "yemeni", "bosnian",
                "black", "white", "asian", "latino", "jewish", "arab", "indian", "african", "nigger",
                "belarusian", "kazakh", "uzbek",
            ],
            "religion": [
                "christian", "catholic", "protestant", "orthodox", "anglican", "baptist", "mormon",
                "jehovah's witness", "judaism", "islam", "hinduism", "buddhism", "sikhism", "shinto",
                "taoism", "atheist", "agnostic", "pagan", "zoroastrianism", "jainism", "bahá'í",
                "scientologist", "rastafarian", "unitarian", "falun gong", "druze", "samaritan",
                "yazidi", "ahmadi", "alawite", "coptic", "animist", "wiccan", "satanist",
                "seventh-day adventist", "muslim", "christian", "jew", "buddhist", "hindu",
            ],
            "gender": [
                "women", "men", "transgender", "non-binary", "intersex", "cisgender", "female", "male",
                "women", "men", "female", "male", "transgender",
            ],
            "sexual orientation": ["lesbian", "gay", "bisexual", "queer", "pansexual", "asexual", "lgbt"],
            "disability": [
                "disabled", "blind", "deaf", "mute", "autistic", "down syndrome", "schizophrenic",
                "bipolar", "mentally ill", "wheelchair user", "paraplegic", "quadriplegic", "dwarf",
                "albino", "epileptic", "diabetic", "hiv positive", "cancer patient", "obese", "amputee",
                "disabled", "retarded", "autistic", "blind", "deaf",
            ],
            "age": ["children", "teenager", "youth", "adult", "senior"],
            "caste": ["dalit", "brahmin", "kshatriya", "vaishya", "shudra"],
            "migration": ["refugee", "immigrant", "migrant", "asylum seeker", "foreigner", "expatriate", "stateless"],
        }
    elif protected_group == "un_zh":
        return {
            "种族": [
                "非洲人", "非裔美国人", "加勒比非洲裔", "阿拉伯人", "亚美尼亚人", "亚洲人", "亚述人",
                "澳大利亚土著", "巴尔干人", "孟加拉人", "巴斯克人", "柏柏尔人", "黑人", "巴西人",
                "英国人", "保加利亚人", "缅甸人", "高加索人", "柬埔寨人", "卡津人", "中国人", "古巴人",
                "捷克人", "丹麦人", "多米尼加人", "荷兰人", "埃及人", "英格兰人", "爱沙尼亚人", "菲律宾人",
                "芬兰人", "法国人", "格鲁吉亚人", "德国人", "希腊人", "吉普赛人/罗姆人", "海地人", "汉族",
                "西班牙裔", "匈牙利人", "冰岛人", "印度人", "土著", "因纽特人", "伊朗人", "伊拉克人",
                "爱尔兰人", "以色列人", "意大利人", "牙买加人", "日本人", "犹太人", "约旦人", "韩国人",
                "库尔德人", "老挝人", "拉丁裔", "黎巴嫩人", "马来人", "毛利人", "墨西哥人", "中东人",
                "蒙古人", "摩洛哥人", "穆斯林", "美洲原住民", "新西兰人", "尼日利亚人", "北欧人",
                "挪威人", "巴基斯坦人", "巴勒斯坦人", "波斯人", "波兰人", "葡萄牙人", "波多黎各人",
                "罗马尼亚人", "俄罗斯人", "沙特阿拉伯人", "苏格兰人", "塞尔维亚人", "新加坡人", "索马里人",
                "南非人", "西班牙人", "斯里兰卡人", "苏丹人", "瑞典人", "瑞士人", "叙利亚人", "台湾人",
                "泰国人", "藏族人", "土耳其人", "乌克兰人", "越南人", "威尔士人", "也门人", "波斯尼亚人",
                "黑人", "白人", "亚洲人", "拉丁裔", "犹太人", "阿拉伯人", "印度人", "非洲人", "黑鬼",
                "白俄罗斯人", "哈萨克人", "乌兹别克人",
            ],
            "宗教": [
                "基督徒", "天主教徒", "新教徒", "东正教徒", "圣公会", "浸礼会", "摩门教徒",
                "耶和华见证人", "犹太教", "伊斯兰教", "印度教", "佛教", "锡克教", "神道教",
                "道教", "无神论者", "不可知论者", "异教徒", "祆教", "耆那教", "巴哈伊教",
                "科学教", "拉斯塔法里教徒", "一神论者", "法轮功", "德鲁兹教徒", "撒马利亚人",
                "雅兹迪教徒", "艾哈迈迪派", "阿拉维派", "科普特教徒", "万物有灵论者", "威卡教徒", "撒旦教徒",
                "基督复临安息日会", "穆斯林", "基督徒", "犹太人", "佛教徒", "印度教徒",
            ],
            "性别": ["女性", "男性", "跨性别", "非二元", "双性人", "顺性别", "女", "男", "女性", "男性", "女", "男", "跨性别"],
            "性取向": ["女同性恋", "男同性恋", "双性恋", "酷儿", "泛性恋", "无性恋", "LGBT", "男同性恋", "女同性恋", "双性恋", "酷儿"],
            "残疾": [
                "残障人士", "盲人", "聋人", "哑巴", "自闭症患者", "唐氏综合症患者", "精神分裂症患者",
                "躁郁症患者", "精神病患者", "轮椅使用者", "截瘫患者", "四肢瘫痪者", "侏儒",
                "白化病患者", "癫痫患者", "糖尿病患者", "艾滋病毒携带者", "癌症患者", "肥胖者", "截肢者",
                "残障人士", "智障", "自闭症患者", "盲人", "聋人",
            ],
            "年龄": ["儿童", "青少年", "青年", "成年人", "老年人"],
            "种姓": ["达利特", "婆罗门", "刹帝利", "吠舍", "首陀罗"],
            "移民状态": ["难民", "移民", "迁徙者", "寻求庇护者", "外国人", "侨民", "无国籍者"],
        }
    elif protected_group == "un_kr":
        return {
            "여성과 소녀": ["여성", "여자", "소녀", "여아", "여학생", "여성 인권옹호자", "여성인권옹호자", "여성 정치인", "여성 언론인", "여성 활동가"],
            "종교적 소수자": ["종교 소수자", "무슬림", "이슬람교도", "유대인", "유대교도", "시크교도", "힌두교도", "불교도", "바하이 신도", "야지디", "아흐마디야 신도", "기독교 소수파", "소수파 기독교인", "소수 종파 신도"],
            "인종 민족 국가적 소수자": ["인종 소수자", "민족 소수자", "국가적 소수자", "국적 소수자", "흑인", "아프리카계", "라틴계", "라티노", "라티나", "라틴엑스", "아시아계", "동남아계", "중국계", "한국계", "일본계", "중동 북아프리카계", "MENA", "아랍인", "쿠르드인", "로마인", "롬인", "팔레스타인인", "로힝야", "유럽 소수 민족", "유럽의 소수 민족"],
            "언어적 소수자": ["언어 소수자", "소수 언어 사용자", "이중언어 화자", "사미어 사용자", "아이누어 사용자", "쿠르드어 사용자", "베르베르어 사용자", "아마지그어 사용자", "카탈루냐어 사용자", "카탈란어 사용자", "한국수어 사용자", "수어 사용자"],
            "이주민 난민 무국적자": ["이주민", "이민자", "이주 노동자", "이주노동자", "난민", "난민 신청자", "망명 신청자", "망명신청자", "국내 실향민", "국내실향민", "IDP", "무국적자", "미등록 이주민", "미등록이주민", "이주 배경 청년", "이주배경 청년"],
            "LGBTIQ+": ["성소수자", "LGBTIQ+", "레즈비언", "게이", "양성애자", "바이섹슈얼", "팬섹슈얼", "범성애자", "무성애자", "에이섹슈얼", "트랜스젠더", "트랜스 여성", "트랜스여성", "트랜스 남성", "트랜스남성", "논바이너리", "비이분법", "젠더 비순응", "젠더비순응", "인터섹스", "퀴어"],
            "원주민": ["원주민", "선주민", "토착민", "아메리카 원주민", "아보리지니", "토레스 해협 섬 주민", "마오리", "사미", "아이누", "인디헤나", "라틴아메리카 원주민", "라틴 아메리카 원주민"],
            "장애인": ["장애인", "지체장애인", "뇌병변장애인", "시각장애인", "청각장애인", "농인", "난청인", "언어장애인", "지적장애인", "발달장애인", "자폐 스펙트럼 당사자", "자폐스펙트럼 당사자", "자폐성 장애인", "학습장애 당사자", "난독증 당사자", "정신장애인", "정신 건강 장애가 있는 사람", "정신건강 장애가 있는 사람", "희귀질환 장애인", "만성질환 장애인"],
            "언론인과 인권옹호자": ["언론인", "기자", "보도진", "편집자", "팩트체커", "여성 언론인", "여성언론인", "인권옹호자", "인권 활동가", "인권활동가", "시민사회 활동가", "시민사회활동가", "법률 지원 활동가", "환경 운동가", "환경운동가", "LGBTIQ+ 인권 활동가", "LGBTIQ+ 인권활동가", "여성 인권 활동가", "여성 인권활동가", "여성인권 활동가", "여성인권활동가"],
        }
    else:
        return {}


def main():
    args = parse_args()
    model = args.model
    json_path = args.data_path
    csv_path = args.input_csv
    output_path = f"{args.output_dir}/{args.dataset}_metric_{model}.json"

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if os.path.exists(output_path):
        json_data_output = []
        with open(output_path, "r", encoding="utf-8") as f:
            text = f.read()

        json_objects = re.findall(r"\{.*?\}(?=\s*{|\s*$)", text, flags=re.DOTALL)
        for obj in json_objects:
            try:
                item = json.loads(obj)
                json_data_output.append(item)
            except Exception as e:
                print("Parse error:", e)
                continue

        start_point = len(json_data_output)
        print(f"The generated samples reloaded, the number of sample is {start_point}s")
    else:
        json_data_output = []
        start_point = 0

    PROTECTED_CLASSES = get_protected_classes(args.protected_group)

    json_data = []
    with open(json_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            json_data.append(sample)

    csv_data = pd.read_csv(csv_path)

    if args.dataset == "implicit":
        text = "post"
        csv_data["label"] = csv_data["class"].map(lambda c: "non-hateful" if c == "not_hate" else "hateful")
        label = "label"
    elif args.dataset == "hatexplain":
        text = "sentence"
        label = "label"
    elif args.dataset == "hatecheck":
        text = "c_testcase"
        label = "t_gold"
    elif args.dataset == "toxicn":
        text = "content"
        label = "label"
    elif args.dataset == "hasoc":
        text = "text"
        label = "class"
    elif args.dataset == "kold":
        text = "comment"
        label = "label"

    metric_weights = {
        "HTC": args.weight_htc,
        "Quotation Faithfulness": args.weight_qf,
        "TGI": args.weight_tgi,
        "Consistency Check": args.weight_cc,
    }

    for i in range(start_point, len(json_data)):
        pred = " "
        prediction = extract_answer_yes(str(json_data[i]["raw"]))
        if prediction == "yes" or prediction == "hateful" or prediction == "hate speech" or prediction == "是，":
            pred = "hateful"
        if prediction == "no" or prediction == "non-hateful" or prediction == "不是，":
            pred = "non-hateful"
        elif prediction == "":
            pred = json_data[i]["label"]

        EXAMPLE = {
            "text": csv_data[text][i],
            "reasoning": json_data[i]["raw"],
            "gold_label": csv_data[label][i],
            "prediction": pred,
        }

        evaluator = ReasoningMetricsEvaluator(
            language=args.lang,
            metric_weights=metric_weights,
            runtime_args=args,
        )
        target_groups = []
        for k, vs in PROTECTED_CLASSES.items():
            for v in vs:
                target_groups.append(k)
                target_groups.append(v)
        per_sample, aggregate = evaluator.evaluate_dataset([EXAMPLE], target_groups)

        gen = {
            "ID": i,
            "input": csv_data[text][i],
            "reasoning": json_data[i]["raw"],
            "label": csv_data[label][i],
            "prediction": pred,
            "flag": json_data[i]["flag"],
        }
        gen.update(per_sample[0])
        write_json(gen, output_path)


if __name__ == "__main__":
    main()
