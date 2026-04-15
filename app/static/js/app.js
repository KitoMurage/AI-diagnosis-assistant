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

// Function to handle sending text to the AI and updating the UI
async function sendMessage(text) {
    if (!text) return;

    const chatBox = document.getElementById('chat-box');

    // 1. Add User's Message to the Chat UI
    const userDiv = document.createElement('div');
    userDiv.className = 'message user-msg';
    userDiv.textContent = text;
    chatBox.appendChild(userDiv);
    
    // Auto-scroll to bottom
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        // 2. Send the request to the specific patient's Consultation ID
        const res = await fetch('/diagnose/' + CONSULT_ID, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await res.json();

        // 3. Add AI's Response to the Chat UI
        const botDiv = document.createElement('div');
        botDiv.className = 'message bot-msg';
        botDiv.innerText = data.response; // Use innerText to keep line breaks
        chatBox.appendChild(botDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        // 4. Update the Patient File (Sidebar) Metrics
        document.getElementById('diag-display').textContent = data.diagnosis;
        document.getElementById('conf-display').textContent = data.confidence;

        // 5. Render Confirmed Symptoms (Green Badges)
        document.getElementById('symptom-list').innerHTML = data.symptoms.map(s => 
            `<span style="background: #27ae60; color: white; padding: 4px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">
                ${s.replace(/_/g, ' ')}
            </span>`
        ).join('');

        // 6. Render Ruled Out Symptoms (Red Badges)
        document.getElementById('denied-list').innerHTML = data.denied.map(s => 
            `<span style="background: #e74c3c; color: white; padding: 4px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">
                ${s.replace(/_/g, ' ')}
            </span>`
        ).join('');

        // 7. Handle Clinical Decision Support: Red Flag Alert
        const redFlagBanner = document.getElementById('red-flag-banner');
        if (data.has_red_flag) {
            redFlagBanner.style.display = 'block';
        } else {
            redFlagBanner.style.display = 'none';
        }

        // 8. Handle Clinical Decision Support: Explainable AI Panel
        const xaiPanel = document.getElementById('xai-panel');
        const xaiList = document.getElementById('xai-list');
        
        if (data.xai_factors && data.xai_factors.length > 0) {
            xaiPanel.style.display = 'block';
            xaiList.innerHTML = ''; // Clear old data
            
            // Loop through the factors and create list items
            data.xai_factors.forEach(factor => {
                let li = document.createElement('li');
                let niceName = factor.symptom.replace(/_/g, ' ');
                // Convert the raw decimal weight to a readable percentage format
                let impactScore = (factor.weight * 100).toFixed(1); 
                
                li.innerHTML = `<strong>${niceName}</strong> (${impactScore}% impact)`;
                li.style.marginBottom = "4px";
                xaiList.appendChild(li);
            });
        } else {
            xaiPanel.style.display = 'none';
        }

    } catch (error) {
        console.error("Error communicating with backend:", error);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message bot-msg';
        errorDiv.style.color = '#e74c3c';
        errorDiv.textContent = "System Error: Cannot connect to the clinical inference engine.";
        chatBox.appendChild(errorDiv);
    }
}

// Helper: Triggered when the user clicks the "Send" button
function manualSend() {
    const inputField = document.getElementById('user-input');
    const text = inputField.value.trim();
    if (text) {
        sendMessage(text);
        inputField.value = ''; // Clear the input field after sending
    }
}

// Helper: Triggered when the user hits the "Enter" key in the text box
function handleKey(event) {
    if (event.key === 'Enter') {
        manualSend();
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