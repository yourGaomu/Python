import os
import requests
import json
import time
from typing import Dict
import pandas as pd

# 定义全局变量
pageSize = 10
pageTotal = 110
poiId = 82527  # 灵隐寺 PoiID

url="https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList"
headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "Content-Length": "1865",
    "Content-Type": "application/json",
    "Origin": "https://you.ctrip.com",
    "Priority": "u=4, i",
    "Referer": "https://you.ctrip.com/",
    "Sec-Ch-Ua": "\"Chromium\";v=\"142\", \"Microsoft Edge\";v=\"142\", \"Not_A Brand\";v=\"99\"",
    "Sec-Ch-Ua-Mobile": "?1",
    "Sec-Ch-Ua-Platform": "\"Android\"",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-Storage-Access": "active",
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Mobile Safari/537.36 Edg/142.0.0.0",
}

cookies = {
    "nfes_isSupportWebP": "1",
    "UBT_VID": "1763646540523.d298KFaysnPw",
    "GUID": "09031086412055754209",
    "_pd": "%7B%22_o%22%3A10%2C%22s%22%3A181%2C%22_s%22%3A1%7D",
    "_ubtstatus": "%7B%22vid%22%3A%221763646540523.d298KFaysnPw%22%2C%22sid%22%3A1%2C%22pvid%22%3A2%2C%22pid%22%3A%22214070%22%7D",
    "Union": "OUID=&AllianceID=66672&SID=1693366&SourceID=&AppID=&OpenID=&exmktID=&createtime=1763646575&Expires=1764251375376",
    "_ga_9BZF483VNQ": "GS2.1.s1763646293$o1$g0$t1763646603$j60$l0$h0",
    "_ga_5DVRDQD429": "GS2.1.s1763646293$o1$g0$t1763646603$j60$l0$h574643842",
    "_ga_B77BES1Z8Z": "GS2.1.s1763646293$o1$g0$t1763646603$j60$l0$h0",
    "_RF1": "219.79.62.100",
    "_RSG": "dqaYma4XuS9CnEwxBzNRm8",
    "_RDG": "28f878319cc44f27021c67bf65e20f226b",
    "_RGUID": "97388449-308a-4d27-8c1b-5fd2fbe8ec4c",
    "_bfa": "1.1763646540523.d298KFaysnPw.1.1763646553323.1763647524891.1.4.290510",
    "MKT_CKID": "1763647546791.ppche.wvlg",
    "_jzqco": "%7C%7C%7C%7C1763647548617%7C1.1985756020.1763647546679.1763647546679.1763647546679.1763647546679.1763647546679.0.0.0.1.1",
}

params = {
    '_fxpcqlniredt': '09031086412055754209',
    'x-traceID': '09031086412055754209-1763646867784-201838',
}

json_data = {
    'arg': {
        'channelType': 2,
        'collapseType': 0,
        'commentTagId': 0,
        'pageIndex': 3,
        'pageSize': 10,
        'poiId': poiId,
        'sourceType': 1,
        'sortType': 3,
        'starType': 0,
    },
    'head': {
        'cid': '09031019117090895670',
        'ctok': '',
        'cver': '1.0',
        'lang': '01',
        'sid': '8888',
        'syscode': '09',
        'auth': '',
        'xsid': '',
        'extension': [],
    },
}

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "评论数据.xlsx")

def start():
    session = requests.Session()
    for page in range(1, pageTotal + 1):
        print(f"正在抓取第 {page} 页评论...")
        time.sleep(1)  # 避免请求过快被封IP
        json_data['arg']['pageIndex'] = page

        result = session.post(
        url="https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList",
        params=params,
        headers=headers,
        json=json_data,
        timeout=10,
        cookies=cookies
    )
        response = result.json()
        datas = response.get('result', {}).get('items') or []
        outpu_content(datas)

def outpu_content(datas):
    # 需要把用户，地点，时间，评论内容保留成xlsx或者csv格式，可以用pandas来处理
    records = []
    for data in datas:
        content = data.get("content") or ""
        ip = data.get("ipLocatedName") or ""
        user_info = data.get("userInfo") or {}
        userName = user_info.get("userNick", "")
        publish_time = data.get("publishTypeTag", "")
        score = data.get("score", 0)
        records.append({"用户": userName, "地点": ip, "时间": publish_time, "内容": content, "评分": score})

    if not records:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    new_df = pd.DataFrame(records, columns=["用户", "地点", "时间", "内容", "评分"])

    if os.path.exists(OUTPUT_FILE):
        try:
            old_df = pd.read_excel(OUTPUT_FILE)
        except Exception:
            old_df = pd.DataFrame(columns=["用户", "地点", "时间", "内容", "评分"])
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_excel(OUTPUT_FILE, index=False)
    


# 主程序入口
if __name__ == "__main__":
    print("开始爬取携程灵隐寺评论数据...")
    start()
    print("评论数据爬取完成，已保存到评论数据.xlsx文件中。")