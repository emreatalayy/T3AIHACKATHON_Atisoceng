import streamlit as st
import json
import requests
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(layout="wide")  # Sayfayı tam geniş yap


CHROMA_PATH = "chroma"
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

def generate_alternatives(original_response, message_index):
    prompt = f"""
    Orijinal yanıt: "{original_response}"
    Lütfen aynı bilgiyi içeren ama farklı şekilde ifade edilmiş iki alternatif yanıt verin. 
    Yanıtlar kısa ve samimi olmalıdır.
    Yanıtlarınızı şu formatta verin:
    Alternatif 1: [İlk alternatif yanıtınız]
    Alternatif 2: [İkinci alternatif yanıtınız]
    """
    
    url = "https://inference2.t3ai.org/v1/completions"
    payload = json.dumps({
        "model": "/home/ubuntu/hackathon_model_2/",
        "prompt": prompt,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 400,
        "repetition_penalty": 1.1,
        "stop_token_ids": [128001, 128009],
        "skip_special_tokens": True
    })
    headers = {
        'Content-Type': 'application/json',
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        alternatives_text = result['choices'][0]['text']
        
        alternatives = alternatives_text.split("Alternatif")
        if len(alternatives) >= 3:
            alt1 = alternatives[1].split(":")[1].strip()
            alt2 = alternatives[2].split(":")[1].strip()
        else:
            raise ValueError("API yanıtı beklenen formatta değil")
        
    except Exception as e:
        st.error(f"")
        alt1, alt2 = generate_manual_alternatives(original_response)
    
    st.session_state.feedback[message_index] = {
        "original": original_response,
        "alternatives": [alt1, alt2]
    }

def generate_manual_alternatives(original_response):
    def replace_words(text):
        replacements = {
            "güzel": "harika",
            "iyi": "mükemmel",
            "büyük": "devasa",
            "küçük": "minik",
            "ilginç": "etkileyici"
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def restructure_sentence(text):
        words = text.split()
        if len(words) > 3:
            mid = len(words) // 2
            return " ".join(words[mid:] + words[:mid])
        return text
    
    alt1 = replace_words(original_response)
    alt2 = restructure_sentence(original_response)
    
    return alt1, alt2

def display_message_with_feedback(message, index):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"like_{index}"):
                    st.session_state.feedback[index] = "like"
            with col2:
                if st.button("👎", key=f"dislike_{index}"):
                    st.session_state.feedback[index] = "dislike"
                    generate_alternatives(message["content"], index)

def save_alternatives_to_file(original, alternative, file_name="alternatives.json"):
    data = {
        "original": original,
        "alternative": alternative
    }
    
    try:
        with open(file_name, "a") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")  # Her kaydın bir alt satıra yazılmasını sağlıyor
        st.success(f"Yanıtlar başarıyla {file_name} dosyasına kaydedildi.")
    except Exception as e:
        st.error(f"Yanıtları kaydederken bir hata oluştu: {str(e)}")

def display_alternatives(message_index):
    if message_index in st.session_state.feedback and isinstance(st.session_state.feedback[message_index], dict):
        alternatives = st.session_state.feedback[message_index]["alternatives"]
        original_response = st.session_state.feedback[message_index]["original"]
        
        for i, alt in enumerate(alternatives):
            st.markdown(f"Alternatif {i+1}: {alt}")
            if st.button(f"Bu Alternatifi Seç", key=f"alt_{message_index}_{i}"):
                st.session_state.messages[message_index]["content"] = alt
                save_alternatives_to_file(original_response, alt)  # Seçilen alternatifi dosyaya kaydet
                st.rerun()
        
        custom_response = st.text_area("Özel yanıtınızı girin:", key=f"custom_{message_index}")
        if st.button("Özel Yanıtı Uygula", key=f"apply_custom_{message_index}"):
            if custom_response.strip():
                st.session_state.messages[message_index]["content"] = custom_response
                save_alternatives_to_file(original_response, custom_response)  # Özel yanıtı dosyaya kaydet
                st.rerun()
            else:
                st.warning("Lütfen özel bir yanıt girin.")

# Sekmeler oluştur
tab1, tab2, tab3 = st.tabs(["T3 Gezgini", "Yakınımdaki Mekanlar", "Harita ve Rota"])

# Birinci sekme - mevcut işleyiş buraya alınacak
with tab1:
    st.title("T3 Gezgini Chatbot")

    # Önce tüm mesajları göster
    for i, message in enumerate(st.session_state.messages):
        display_message_with_feedback(message, i)

    # Sonra, gerekirse alternatifleri göster
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant" and i in st.session_state.feedback:
            if isinstance(st.session_state.feedback[i], dict):
                display_alternatives(i)

    # Yeni mesaj girişi
    if query_text := st.chat_input("T3 Gezginine bir mesaj gönderin"):
        st.session_state.messages.append({"role": "user", "content": query_text})
        
        results = db.similarity_search_with_relevance_scores(query_text, k=3)

        PROMPT_TEMPLATE = """
        Sen Antalya'yı ziyaret edecek bir turist için arkadaşça bir tonda konuşan yardımcı bir asistansın. Samimi ve rehber edici bir dil kullanarak soruları cevapla.

        Kullanıcının sorusu: {question}
        Turistin ilgi alanları ve mevcut durumu: {context}

        ---
        Soruyu yanıtla ve turist için samimi bir öneride bulun.
        """

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        url = "https://inference2.t3ai.org/v1/completions"
        payload = json.dumps({
            "model": "/home/ubuntu/hackathon_model_2/",
            "prompt": prompt,
            "temperature": 0.01,
            "top_p": 0.95,
            "max_tokens": 800,
            "repetition_penalty": 1.1,
            "stop_token_ids": [128001, 128009],
            "skip_special_tokens": True
        })
        headers = {
            'Content-Type': 'application/json',
        }

        response = requests.post(url, headers=headers, data=payload)
        pretty_response = json.loads(response.text)

        assistant_response = pretty_response['choices'][0]['text']
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        display_message_with_feedback(st.session_state.messages[-1], len(st.session_state.messages) - 1)

        st.rerun()

# İkinci sekme - şu anda boş
with tab2:
    search_api_url = "https://api.foursquare.com/v3/places/search"
    details_api_url = "https://api.foursquare.com/v3/places/{place_id}/photos"
    api_key = "fsq3ZqNV3H8BusXm8GdudH5zNDCcLcQ9jnwajNYfApQNAgg="  # API anahtarı kodda yer alıyor

    # Kullanıcıdan sorgu parametrelerini alalım
    query = st.text_input("Yer Araması İçin Anahtar Kelime Girin:", value="kafe")
    location = st.text_input("Mekan İsmi Girin (örn. Antalya, İstanbul):", value="Antalya")

    # Eğer kullanıcı bir parametre girdiyse ve sorgu başlat butonuna tıkladıysa
    if st.button("Sorgu Başlat"):
        if api_key:
            try:
                headers = {
                    "Accept": "application/json",
                    "Authorization": api_key  # API anahtarını başlıkta kullanıyoruz
                }

                # Parametreleri oluştur
                params = {
                    "query": query,
                    "near": location,  # Enlem-boylam yerine "near" parametresiyle mekan adı kullanıyoruz
                    "limit": 5  # İlk 5 sonucu döndürmek için limit koyuyoruz
                }

                # Mekan araması için API'ye GET isteği gönder
                search_response = requests.get(search_api_url, headers=headers, params=params)
                search_response.raise_for_status()  # Hata varsa exception fırlatır

                # Yanıtı JSON olarak işle
                search_data = search_response.json()

                # API yanıtını kullanıcıya göster
                st.write("İlk 5 Mekan:")

                # API yanıtından ilk 5 mekanı al
                places = search_data.get("results", [])

                if places:
                    # Her bir mekanı işleyelim
                    for place in places:
                        name = place.get('name', 'İsim Bulunamadı')
                        address = place.get('location', {}).get('address', 'Adres Bulunamadı')

                        # Mekanın ID'sini al
                        place_id = place.get('fsq_id')

                        # Fotoğrafları almak için mekan detaylarına API çağrısı yap
                        photos_response = requests.get(details_api_url.format(place_id=place_id), headers=headers)
                        photos_response.raise_for_status()

                        # Fotoğraf yanıtını al
                        photos_data = photos_response.json()

                        if photos_data:
                            # İlk fotoğrafın URL'sini oluştur
                            photo_prefix = photos_data[0].get('prefix', '')
                            photo_suffix = photos_data[0].get('suffix', '')
                            photo_url = f"{photo_prefix}original{photo_suffix}"
                        else:
                            # Fotoğraf yoksa varsayılan bir görsel kullan
                            photo_url = "https://via.placeholder.com/150"

                        # Mekanın bilgilerini göster
                        st.image(photo_url, caption=name, width=200)
                        st.write(f"*Adres:* {address}")

                else:
                    st.write("Hiçbir yer bulunamadı.")

            except requests.exceptions.RequestException as e:
                st.error(f"API isteğinde bir hata oluştu: {e}")
        else:
            st.warning("Lütfen geçerli bir Foursquare API anahtarı girin.")


with tab3:

    import re  # HTML etiketlerini temizlemek için düzenli ifadeler modülü

    st.subheader("Harita Sayfası")

    # Varsayılan konum: Antalya
    default_lat = 36.9400971
    default_lon = 30.8171021

    # Kullanıcının konumunu otomatik almak
    location = streamlit_geolocation()

    # Google Maps API Anahtarı
    GOOGLE_MAPS_API_KEY = "AIzaSyDz9QBazwx70Kt19VrxyYQAtgisK1MNyv0"

    # Daha geniş kolonlar için oranları büyük tutuyoruz
    col1, col2 = st.columns([5, 4])  # Sol (harita) için daha büyük kolon, sağ (chatbot) için geniş kolon

    with col1:
        # Harita bölümü
        st.write("### Harita")

        # İlk olarak varsayılan konumla haritayı göster
        lat = default_lat
        lon = default_lon
        st.write(f"Varsayılan konum: Enlem {lat}, Boylam {lon}")

        # Google Maps Embed API ile harita gösterme
        map_url = f"https://www.google.com/maps/embed/v1/view?key={GOOGLE_MAPS_API_KEY}&center={lat},{lon}&zoom=14&maptype=satellite"
        st.markdown(f"""
        <iframe
            width="100%"
            height="600"
            frameborder="0" style="border:0"
            src="{map_url}" allowfullscreen>
        </iframe>
        """, unsafe_allow_html=True)

        # Eğer kullanıcı konumu güncellerse, haritayı yeni konuma göre güncelle
        if location:
            lat = location['latitude']
            lon = location['longitude']
            st.write(f"Konumunuz güncellendi: Enlem {lat}, Boylam {lon}")

            # Yeni konumu kullanarak haritayı güncelle
            map_url = f"https://www.google.com/maps/embed/v1/view?key={GOOGLE_MAPS_API_KEY}&center={lat},{lon}&zoom=14&maptype=satellite"
            st.markdown(f"""
            <iframe
                width="100%"
                height="600"
                frameborder="0" style="border:0"
                src="{map_url}" allowfullscreen>
            </iframe>
            """, unsafe_allow_html=True)

    # CSS ile chat kutusuna kaydırma çubuğu ekliyoruz
    st.markdown("""
        <style>
        .chatbox {
            max-height: 300px;
            overflow-y: scroll;
            padding-right: 15px;
        }
        </style>
        """, unsafe_allow_html=True)

    with col2:
        # Chatbot bölümü
        st.write("### Chatbot")

        # Google API key
        API_KEY = 'AIzaSyDz9QBazwx70Kt19VrxyYQAtgisK1MNyv0'

        def get_directions(origin, destination, mode="driving", language="tr"):
            url = f'https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&mode={mode}&language={language}&key={API_KEY}'
            response = requests.get(url)
            directions = response.json()

            if directions['status'] == 'OK':
                # Get total distance and duration
                total_distance = directions['routes'][0]['legs'][0]['distance']['text']
                total_duration = directions['routes'][0]['legs'][0]['duration']['text']
                
                steps_info = f"Toplam Mesafe: {total_distance}\nToplam Süre: {total_duration}\n\nAdım adım yol tarifi:\n"
                
                # HTML etiketlerini temizleme fonksiyonu
                def clean_html(raw_html):
                    clean_text = re.sub(r'<.*?>', '', raw_html)
                    return clean_text

                # Get step-by-step directions
                for leg in directions['routes'][0]['legs']:
                    for step in leg['steps']:
                        instruction = clean_html(step['html_instructions'])  # HTML etiketlerinden arındır
                        distance = step['distance']['text']
                        duration = step['duration']['text']
                        steps_info += f"Yol Tarifi: {instruction}\nMesafe: {distance}\nSüre: {duration}\n\n"
                
                return steps_info
            else:
                return f"Hata: {directions['status']}"

        # Session state for messages
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat interface
        messages = st.container()

        # Kullanıcıdan giriş alınıyor
        if prompt := st.chat_input("Gitmek istediğiniz yeri yazın"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Kullanıcıdan hedef lokasyon al ve API ile yol tarifi al
            origin = 'ANFAŞ Antalya'  # Varsayılan olarak başlangıç noktası
            destination = prompt  # Kullanıcının girdiği lokasyon hedef olarak alınır
            
            assistant_reply = get_directions(origin, destination)
            
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        # Mesajları göster
        st.write('<div class="chatbox">', unsafe_allow_html=True)  # Chatbox başı
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["content"])
            else:
                messages.chat_message("assistant").write(message["content"])
        st.write('</div>', unsafe_allow_html=True)  # Chatbox sonu
