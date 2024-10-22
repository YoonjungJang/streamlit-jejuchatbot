import os
import streamlit as st

from PIL import Image, ImageOps
from io import BytesIO
import requests
import folium

import streamlit.components.v1 as components
import time
import re



# 이미지 크기 조정 함수 정의
def resize_image(image_url, target_width=150, target_height=150):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img_resized = img.resize((target_width, target_height), Image.LANCZOS)
    return img_resized

def resize_emoji(emoji, font_size=50):
   
    html_code = f"<div style='text-align: center;'><span style='font-size: {font_size}px;'>{emoji}</span></div>"
    return html_code

# 이미지 출력 함수 정의
def display_image(img, caption):
    st.image(img, caption=caption, width=150)

    


# 페이지 변경 함수
def change_page(page):
    st.session_state.page = page
    st.experimental_rerun()




# 별점을 HTML로 변환하는 함수
def render_stars(rating):
    full_stars = int(rating)  # 전체 별 개수
    half_star = (rating - full_stars) >= 0.5  # 반 별이 필요한지
    empty_stars = 5 - full_stars - int(half_star)  # 빈 별 개수

    # 별 아이콘만 살짝 위로 올리기 위한 CSS 적용
    stars_html = '<div style="font-size: 18px; color: orange; margin-top: -10px; margin-left: -8px;">'
    stars_html += '<i class="fas fa-star"></i> ' * full_stars
    if half_star:
        stars_html += '<i class="fas fa-star-half-alt"></i> '
    stars_html += '<i class="far fa-star"></i> ' * empty_stars
    stars_html += '</div>'

    return stars_html

def render_dollars(d_rating):
    full_dollars = int(d_rating)  # 전체 달러의 수
    half_dollar = int((d_rating - full_dollars) >= 0.5)  # 반 달러의 유무
    empty_dollars = 4 - full_dollars - half_dollar  # 빈 달러의 수

    # 달러 아이콘을 포함하는 HTML 문자열 생성
    
    dollars_html = '<div style="width: 200px; height: 200px; font-size: 22px; color: green; margin-top: -13px; margin-left: -6px;">'
    dollars_html += '<i class="fas fa-dollar-sign"></i> ' * full_dollars
    
    if half_dollar:
        dollars_html += '<i class="fas fa-dollar-sign" style="opacity: 0.5;"></i> '
    dollars_html += '<i class="fas fa-dollar-sign" style="opacity: 0.3;"></i> ' * empty_dollars
    dollars_html += '</div>'

    return dollars_html

# 마커 생성 함수
def create_marker(lat, lon, name, address, gmap_id, color='red'):
    html = f"""
    <div style='text-align: left; color: black; font-size: 14px;'>
        {name}<br>{address}<br>
    </div>
    """
    iframe = folium.IFrame(html, width=250, height=100)
    popup = folium.Popup(iframe, max_width=265)
    return folium.Marker(
        [lat, lon],
        popup=popup,
        tooltip=f"{name}",
        icon=folium.Icon(color=color)
    )
    
# 정보 업데이트 함수
# def update_info_container(gmap_id,merged_dict):
#     with sql.connect(server_hostname=HOST, http_path=HTTP_PATH, access_token=PERSONAL_ACCESS_TOKEN) as conn:
#         with conn.cursor() as cursor:
#             if not gmap_id:
#                 return
            
#             query = f"""
#             SELECT name2, first_main_category2, region2, avg_rating2, price2, city2, hash_tag2,url2 
#             FROM `hive_metastore`.`streamlit`.`gmap_id2_info`
#             WHERE gmap_id2 = '{gmap_id}'
#             """
            

#             cursor.execute(query)
#             result = cursor.fetchone() 
#             prob = merged_dict.get(str(gmap_id), None)
#             if result == None:
#                 query = f"""
#                     SELECT name1, first_main_category1, region1, avg_rating1, price1, city1, hash_tag1,url1
#                     FROM `hive_metastore`.`streamlit`.`gmap_id1_info`
#                     WHERE gmap_id1 = '{gmap_id}'
#                     """
#                 cursor.execute(query)
#                 result = cursor.fetchone()
#             if result:
#                 name, first_main_category, region, avg_rating, price,city, hash_tag,url = result
#                 category_emoji = get_category_emoji(first_main_category)
#                 st.write_stream(stream_data(f'**장소명** :  [{name}]({url})'))
#                 st.write_stream(stream_data(f'**카테고리** :  {category_emoji}{ first_main_category}'))
#                 st.write_stream(stream_data(f'**지역명** :  🌐{region}'))
#                 st.write_stream(stream_data(f'**도시명** :  🏙️{city}'))
                
#                 col3, col4 = st.columns([2,9])
#                 with col3:
#                     st.write(f'**별점** :') 
#                     st.write(f'**가격** :')
#                 with col4:
#                     # 별점
#                     rating = avg_rating
#                     star_html = f"""
#                         <head>
#                         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
#                         </head>{render_stars(rating)}
#                         """
#                     components.html(star_html, height=20)
                
#                     #달러
#                     try:
#                         price=int(price)
#                     except:
#                         price=0
#                     dollars_html = f"""
#                         <head>
#                         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
#                         </head>{render_dollars(price)}
#                         """
#                     components.html(dollars_html, height=22)
#                 try:
#                     if prob == None:
#                         pass
#                     else:
#                         st.write_stream(stream_data(f'**Prob** :  {prob}'))
                    
#                 except:
#                     pass
                
#                 #해시태그 (색상 변경)
#                 hashtags = ' '.join([f'#{tag}' for tag in hash_tag])
#                 hashtag_html = f"<p style='color:orange;'>{hashtags}</p>"
#                 st.markdown(hashtag_html, unsafe_allow_html=True)
                
#             else:
#                 st.write("선택된 장소의 정보를 찾을 수 없습니다.")

                
def create_emoji_marker(lat, lon, name, address, gmap_id, first_main_category, color='red',g_rank='',url=''):
    # HTML 콘텐츠에 이모지 포함
    category_emoji = get_category_emoji(first_main_category)
    html = f"""
        <div style='text-align: left; font-size: 14px;'>
            <strong style='color: #0078AA;'>{g_rank}{name}</strong><br>
            <span style='font-size: 16px; color: #333;'>{category_emoji}</span> 
            <span style='color: #555;'>⭐{4.3}</span>
        </div>
        """
    iframe = folium.IFrame(html, width=300, height=100)
    popup = folium.Popup(iframe, min_width=300, max_width=300)
    marker = folium.Marker(
        [lat, lon],
        popup=popup,
        tooltip=f"{name}{category_emoji}",
        icon=folium.Icon(color=color)
    )
    return marker.add_child(folium.Popup(html, show=True))

# 카테고리별 이모지 맵핑 함수
def get_category_emoji(category):
    category_emojis = {
        "Corporate & Office" : "🏢",
        "Entertainment & Recreation" : "🎡",
        "Finance & Legal" : "💰",
        "Food & Beverage" : "🍔",
        "Healthcare" : "🩺",
        "Nature & Outdoor" : "🌳",
        "Other" : "🏷️", 
        "Religious" : "⛪", 
        "Residential" : "🏠", 
        "Retail" : "🛍️", 
        "Service" : "🛎️"
    }
    return category_emojis.get(category, "🏷️")  # 기본 이모지로 '🏷️' 사용

# def preprocess_data(data):
#     # Split the data into lines
#     lines = data.strip().split("\n")
    
#     # Extract summary sentences
#     summary = [line.strip() for line in lines if line.strip().startswith("1.") or line.strip().startswith("2.") or line.strip().startswith("3.")]
    
#     # Extract keywords
#     keyword_section_found = False
#     keywords = []
    
#     for line in lines:
#         line = line.strip()
#         if line.startswith("10 key keywords:") or line.startswith("주요 키워드:") or line.startswith("키워드:")or line.startswith("keywords:"):
#             keyword_section_found = True
#             continue
        
#         if keyword_section_found:
#             if line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4.") or line.startswith("5.") or line.startswith("6.") or line.startswith("7.") or line.startswith("8.") or line.startswith("9.") or line.startswith("10."):
#                 keywords.append(line.split(" ")[1])  # Get the word after the number and period
def preprocess_data(data):
    lines = data.strip().split("\n")
    summary = [line.strip() for line in lines if line.strip().startswith("1.") or line.strip().startswith("2.") or line.strip().startswith("3.")]

    keywords = []
    keyword_section_found = False
    
    keyword_start_pattern = re.compile(r"^(10 key keywords:|주요 키워드:|키워드:|keywords:)", re.IGNORECASE)
    
    for line in lines:
        line = line.strip()
        if keyword_start_pattern.match(line):
            keyword_section_found = True
            continue
        
        if keyword_section_found:
            # 숫자와 점 다음의 모든 텍스트를 추출
            match = re.match(r"\d+\.\s*(.*)", line)
            if match:
                keywords.append(match.group(1))  # Get the entire text after the number and period

    return summary, keywords
    
    return summary, keywords
#글씨 적어지는 함수
def stream_data(_LOREM_IPSUM):
    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.02)
# Example of how to use the function in the Streamlit app
def review_summary(input_data):
    #st.write(input_data)
    summary, keywords = preprocess_data(input_data)
    
    # Display the summary in Streamlit with emojis
    st.write_stream(stream_data(f"💬 {summary[0]}"))
    st.write_stream(stream_data(f"💬 {summary[1]}"))
    st.write_stream(stream_data(f"💬 {summary[2]}"))
    st.write('-----------')
    # Display the keywords with emojis
    st.subheader("🔑 **10 Keywords**")
    col_li1,col_li2 = st.columns([1,1])
    
    for idx, keyword in enumerate(keywords, start=1):
        iidx = idx-1
        if idx < 6:
            with col_li1: 
                st.write_stream(stream_data(f"{iidx}️⃣ **{keyword}**"))
        else:
            with col_li2: 
                st.write_stream(stream_data(f"{iidx}️⃣ **{keyword}**"))
                
#리뷰요약 출력 버전 2                
def review_summary2(input_data1,input_data2):
    #st.write(input_data)
    summary1, keywords1 = preprocess_data(input_data1)
    summary2, keywords2 = preprocess_data(input_data2)
    col_gmap1,col_gmap2 = st.columns([5,5])
    keywords = [keywords1,keywords2]
    # 선택된 위치의 리뷰 표시
    with col_gmap1:
        st.subheader("📜선택된 위치 리뷰")
        # Display the summary in Streamlit with emojis
        st.write_stream(stream_data(f"💬 {summary1[0]}"))
        st.write_stream(stream_data(f"💬 {summary1[1]}"))
        st.write_stream(stream_data(f"💬 {summary1[2]}"))
    with col_gmap2:
        st.subheader("📜추천 위치 리뷰")
        # Display the summary in Streamlit with emojis
        st.write_stream(stream_data(f"💬 {summary2[0]}"))
        st.write_stream(stream_data(f"💬 {summary2[1]}"))
        st.write_stream(stream_data(f"💬 {summary2[2]}"))
    a,b  = st.columns([5,5])
    with a:
        st.write('-----------')
    with b:
        st.write('-----------')
    col_gmap3,col_gmap4 = st.columns([5,5])
    # Display he keywords with emojis
    for col_idx,i in enumerate([col_gmap3,col_gmap4]):
        with i:
            st.subheader("🔑 **10 Key Keywords**")
            
            col_li1,col_li2 = st.columns([1,1])
            for idx, keyword in enumerate(keywords[col_idx], start=1):
                iidx = idx-1
                if idx < 6:
                    with col_li1: 
                        st.write_stream(stream_data(f"{iidx}️⃣ **{keyword}**"))
                else:
                    with col_li2: 
                        st.write_stream(stream_data(f"{iidx}️⃣ **{keyword}**"))
            
def create_gmap_id_prob_dict(gmap_id2_values, prob,rank):
    """
    gmap_id2_values와 hybrid_prob 리스트를 이용해 딕셔너리를 생성하는 함수.

    Parameters:
    - gmap_id2_values (list): gmap_id2 값들의 리스트.
    - hybrid_prob (list): gmap_id2에 해당하는 확률 값들의 리스트.

    Returns:
    - dict: gmap_id2를 키로, hybrid_prob를 값으로 가지는 딕셔너리.
    """
    # gmap_id2와 hybrid_prob를 딕셔너리로 매핑
    gmap_id_prob_dict = {gmap_id2_values[i]: prob[i] for i in range(len(gmap_id2_values))}
    gmap_id_rank_dict = {gmap_id2_values[i]: rank[i] for i in range(len(gmap_id2_values))}
    
    return gmap_id_prob_dict,gmap_id_rank_dict

