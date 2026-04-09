const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isMicOn = false;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = true; 
    recognition.interimResults = true; 

    recognition.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }
        
        document.getElementById("user-input").value = interimTranscript;

        if (finalTranscript.trim() !== '') {
            document.getElementById("user-input").value = "";
            sendMessage(finalTranscript);
        }
    };

    recognition.onend = () => {
        if (isMicOn) {
            console.log("Auto-restarting microphone...");
            recognition.start();
        }
    };
}

function toggleMic() {
    if (!recognition) return alert("Browser not supported");
    
    const btn = document.getElementById("mic-btn");
    const input = document.getElementById("user-input");

    if (isMicOn) {
        isMicOn = false;
        recognition.stop();
        btn.classList.remove("listening");
        input.placeholder = "Mic off. Click icon to start...";
    } else {
        isMicOn = true;
        recognition.start();
        btn.classList.add("listening");
        input.placeholder = "Listening continuously... Speak freely.";
    }
}

function manualSend() {
    const input = document.getElementById("user-input");
    const message = input.value.trim();
    if (message) {
        input.value = "";
        sendMessage(message);
    }
}

async function sendMessage(text) {
    addMessage(text, "user-msg");

    try {
        const res = await fetch('/diagnose', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: text})
        });
        const data = await res.json();
        
        addMessage(data.response, "bot-msg");
        updateSidebar(data);
    } catch (error) {
        console.error("Error:", error);
    }
}

function addMessage(text, className) {
    const div = document.createElement("div");
    div.className = `message ${className}`;
    div.innerText = text;
    const box = document.getElementById("chat-box");
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}

function updateSidebar(data) {
    document.getElementById("diag-display").innerText = data.diagnosis;
    document.getElementById("conf-display").innerText = data.confidence;
    
    const list = document.getElementById("symptom-list");
    list.innerHTML = "";
    data.symptoms.forEach(s => {
        const tag = document.createElement("span");
        tag.className = "symptom-tag"; tag.innerText = s; list.appendChild(tag);
    });

    const deniedList = document.getElementById("denied-list");
    deniedList.innerHTML = "";
    data.denied.forEach(s => {
        const tag = document.createElement("span");
        tag.className = "symptom-tag denied-tag"; tag.innerText = s; deniedList.appendChild(tag);
    });
}

async function resetSession() {
    await fetch('/reset', {method: 'POST'});
    location.reload();
}

function handleKey(e) { if (e.key === 'Enter') manualSend(); }