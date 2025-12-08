# -*- coding: utf-8 -*- 
import requests
from lxml import etree
from typing import Optional
import sys

# 确保输出编码为 UTF-8，避免中文乱码
sys.stdout.reconfigure(encoding='utf-8')
# 定义一个函数来发送请求并解析HTML响应
def fetch_and_parse(url: str, data: Optional[dict] = None, headers: Optional[dict] = None, method: str = "GET", timeout: int = 10) -> etree.Element:
    # 默认使用提供的浏览器头信息，模拟真实请求
    if headers is None:
        headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-encoding": "gzip, deflate",  # 移除 br, zstd 以避免解码问题
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "cache-control": "max-age=0",
            "cookie": "bid=oIuffUqSdKQ; ap_v=0,6.0; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1762848245%2C%22https%3A%2F%2Flink.csdn.net%2F%3Ffrom_id%3D109739116%26target%3Dhttps%3A%2F%2Fmovie.douban.com%2Ftop250%22%5D; _pk_id.100001.4cf6=dc92cbf94d62a29a.1762848245.; _pk_ses.100001.4cf6=1; __yadk_uid=2oG5R549I0tq95w6xLLUzWLC8hD8k6Ml; __utma=30149280.1702079123.1762848249.1762848249.1762848249.1; __utmb=30149280.0.10.1762848249; __utmc=30149280; __utmz=30149280.1762848249.1.1.utmcsr=link.csdn.net|utmccn=(referral)|utmcmd=referral|utmcct=/; __utma=223695111.1323688569.1762848249.1762848249.1762848249.1; __utmb=223695111.0.10.1762848249; __utmc=223695111; __utmz=223695111.1762848249.1.1.utmcsr=link.csdn.net|utmccn=(referral)|utmcmd=referral|utmcct=/",
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
    # 根据是否有 data 自动选择方法，但允许手动指定
    if method.upper() == "POST":
        response = requests.post(url, data=data, headers=headers, timeout=timeout)
    else:
        response = requests.get(url, params=data, headers=headers, timeout=timeout)
    print("状态码:", response.status_code)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 错误: {e}")
        return None  # 或返回空树
    parser = etree.HTMLParser()
    tree = etree.fromstring(response.content, parser)
    return tree

# 解释html里面的内容
def xpath2html(tree: etree.Element) -> None:
    global title
    movies  = tree.xpath("//ol/li")
    for movie in movies:
        # 如果某一项没空就用""空字符串代替
        cn_title = movie.xpath(".//span[@class='title'][1]/text()") or [""]
        eg_title = movie.xpath(".//span[@class='title'][2]/text()") or [""]
        a_url= movie.xpath(".//div[@class='hd']/a/@href") or [""]
        rating = movie.xpath(".//span[@class='rating_num'][1]/text()") or ["0"]
        if eg_title[0]:
            title = f"{cn_title[0]}{eg_title[0]}"
        rank = movie.xpath(".//em/text()") or [""]
        print(f"排名: {rank[0]}, 电影名: {title}, 链接: {a_url[0]}, 评分: {rating[0]}")

tree = fetch_and_parse("https://movie.douban.com/top250")
if tree is not None:
    xpath2html(tree)
else:
    print("无法解析页面")



