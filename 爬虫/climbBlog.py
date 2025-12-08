import requests
from lxml import etree
from typing import Optional, Dict, List
import os
import json  # 添加 json 模块用于解析

session = requests.Session()

target_url = "http://38.14.249.194:2444/blogWeb/article/list"
# 需要自己去拼接
articles_url = "http://38.14.249.194:2444/blog/#/article/"

articles_content_url = "http://38.14.249.194:2444/blogWeb/article/detail"


def request_fetch_and_parse(url: str, headers: Optional[Dict[str, str]] = None) -> etree.Element:
    current = 1
    size =10
    data = {
        "current": current,
        "size": size
    }
    response = session.post(url, json=data, headers=headers)
    data = response.json()  # 解析为 JSON
    # 这里可以根据需要处理 data 并构建 etree.Element
    return data

def extract_data_to_dict(data: Dict) -> Dict[str, list]:
    articles = data.get("data", [])
    result = {
        "category_id": [],
        "category_name": [],
        "title": [],
        "summary": [],
        "id": [],
        "article_url": []
    }
    for item in articles:
        category_id = item.get("category", []).get("id", "")
        category_name = item.get("category", []).get("name", "")
        title = item.get("title", "")
        id = item.get("id", "")
        summary = item.get("summary", "")
        # 拼接文章链接,并且文章id不为空
        if id:
            article_url = f"{articles_url}{id}"
        else:
            article_url = ""
        result["category_id"].append(category_id)
        result["category_name"].append(category_name)
        result["title"].append(title)
        result["summary"].append(summary)
        result["article_url"].append(article_url)
        result["id"].append(id)
    return result

def get_article_content(article_id: list) -> None:
    for articleid in article_id:
        response = session.post(articles_content_url, json={"articleId": articleid})
        if response.status_code == 200:
            resp = response.json()
             # 处理文章内容
            content = resp.get("data", {}).get("content", "")
             # 处理文章标题
            title = resp.get("data", {}).get("title", "")
            # 把内容存入当前目录的文件
            os.makedirs("./data", exist_ok=True)
            try:
                with open(f"./data/article_{articleid}_{title}.html", "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                print(f"写入文件失败: {e}")  
        else:
            print(f"无法获取文章内容，状态码: {response.status_code}")


# 主程序
if __name__ == "__main__":
    headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-encoding": "gzip, deflate",  # 移除 br, zstd 以避免解码问题
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "cache-control": "max-age=0",
            "priority": "u=0, i",
            "referer": "https://link.csdn.net/?from_id=109739116&target=https%3A%2F%2Fmovie.douban.com%2Ftop250",
            "sec-ch-ua": '"Chromium";v="142", "Microsoft Edge";v="142", "Not_A Brand";v="99"',
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": '"Android"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "cross-site",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Mobile Safari/537.36 Edg/142.0.0.0"
        }
    # 输入对应的url链接了之后，进行post请求获取数据，并且已经进行了数据的json处理
    data = request_fetch_and_parse(target_url, headers=headers)
    # 解析 data 并提取所需信息
    data_dict = extract_data_to_dict(data)
    get_article_content(data_dict['id'])

