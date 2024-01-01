css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message.bot-upload-warning {
    background-color: #FF0000;  # Change this to the desired background color
}
.chat-message.bot-upload-warning .message {
    font-weight: bold;
    font-size: 1.5em;  # Adjust the font size as desired
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
# #MainMenu {visibility: hidden;}
footer {visibility: hidden;}
# header {visibility: hidden;}
'''

bot_template = '''
<div class="chat-message bot {{EXTRA_CLASS}}">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/TPPRFRD/fotor-2023-6-8-5-37-45.png" alt="fotor-2023-6-8-5-37-45" border="0">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
