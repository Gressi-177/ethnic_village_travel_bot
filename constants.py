# model_file_path = "model/vinallama-7b-chat_q5_0.gguf"
model_file_path = "model/vinallama-2.7b-chat_q5_0.gguf"
db_path = "data/chroma"
data_path = "data"

threshold = 0.5
min_score = -10
max_score = 0

API_KEY = "AIzaSyBNcRbg5tHhewhbv4QZMrqtkBVWtgRbBsc"

ethnic_groups = [
    "Ba Na", "Chăm", "Bru Vân Kiều", "Bố Y", "Brâu", "Chơ Ro", "Chu Ru",
    "Chứt", "Cờ Lao", "Cơ Tu", "Co", "Cống", "Dao", "Ê Đê", "Gia Rai",
    "Giáy", "Gié Triêng", "H Mông", "Hà Nhì", "Hrê", "K Ho", "Kháng",
    "Khơ Me", "Khơ Mú", "La Chí", "La Ha", "La Hủ", "Lào", "Lô Lô", "Mảng",
    "Mường", "Ngái", "Nùng", "Ơ Đu", "Pà Thẻn", "Phù Lá", "Pu Péo", "Ra Glai", 
    "Rơ Măm", "Sán Chay", "Sán Dìu", "Si La", "Tà Ôi", "Tày", "Thái", "Thổ", 
    "Xinh Mun", "Xơ Đăng", "Xtiêng"
]

ETHNIC_MAP = {
    "Mường": [
        "muong", "mương", "mường", "muơng", "mươngg", "mườnng", "muongg",
        "mường ", "mườ ng", "mươ ng", "mu ơng"
    ],
    "Ba Na": [
        "bana", "bà na", "ba-na", "b na", "ba nà", "bàna", "banna", "baana"
    ],
    "Brâu": [
        "brau", "bráu", "brau ", "br au", "bràu", "brâu", "braù", "b'rau"
    ],
    "Bru Vân Kiều": [
        "bru", "van kieu", "bru vankieu", "bru van kiêu", "bru van kiêù", "bru vaan kiêù",
        "bru vân kiêu", "b ru vân kiều", "bru vân kieu", "bru-van-kieu"
    ],
    "Chăm": [
        "cham", "chấm", "chàm", "chaam", "cham ", "c hăm", "chãm", "chăm ", "ch àm"
    ],
    "Chu ru": [
        "chu ru", "churu", "ch u ru", "chu-ru", "churu ", "chu rù", "chu rư"
    ],
    "Cơ Tu": [
        "co tu", "cotu", "cơt u", "cơ-tu", "cờ tu", "co tú", "cơ tù"
    ],
    # "Co": [
    #     "co", "cô", "coo", "cò", "cỏ", "co ", "c ô"
    # ],
    "Cống": [
        "cong", "cóng", "cống ", "c ống", "công", "c0ng"
    ],
    "Tày": [
        "tay", "tầy", "tày ", "t ày", "tà y", "tài", "tầy ", "taay"
    ],
    "Sán Dìu": ["sán dìu", "san diu"]
}