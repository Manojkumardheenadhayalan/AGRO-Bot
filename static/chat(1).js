document.addEventListener('DOMContentLoaded', () => {
  // Your JavaScript code here
  // Retrieve the user's name (replace 'userName' with your actual variable)
  
  const userName = document.getElementById('name').value;
   // Replace this with your variable or backend retrieval
  
  const chatInput = document.querySelector("#chat-input");
  const sendButton = document.querySelector("#send-btn");
  const chatContainer = document.querySelector(".chat-container");
  const themeButton = document.querySelector("#theme-btn");
  const deleteButton = document.querySelector("#delete-btn");
  const menuIcon = document.querySelector("#menu-icon"); // Add menu icon reference
  const API_URL = "/assist";
  var currentHour = new Date().getHours();

  function getGreeting(currentHour) {
      if (currentHour >= 5 && currentHour < 12) {
          return ' Good morning ';
      } else if (currentHour >= 12 && currentHour < 17) {
          return ' Good afternoon ';
      } else if (currentHour >= 17 && currentHour < 21) {
          return ' Good evening ';
      } else {
          return '';
      }
  }

  
  // Create a greeting message
  const greetingMessage = `Hello, ${userName}!${getGreeting(currentHour)}. How can I assist you today?`;

  var md = window.markdownit();

  // Sidebar toggler
  const sidebar = document.querySelector('.sidebar');
  const toggleButton = document.querySelector('.toggle-sidebar-button');
  const hideSidebarButton = document.querySelector('.hide-sidebar-button');

  toggleButton.addEventListener('click', () => {
    console.log('Toggle button clicked'); // Add this line
    sidebar.classList.toggle('active');
  });

  hideSidebarButton.addEventListener('click', () => {
    console.log('Hide button clicked'); // Add this line
    sidebar.classList.remove('active');
  });

  sendButton.addEventListener("click", () => {
    console.log("Button clicked"); // Add this line
    handleOutgoingChat();
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleOutgoingChat();
    }
  });

  const handleOutgoingChat = async () => {
    const userText = chatInput.value.trim();
    if (!userText) return;

    appendUserMessage(userText);
    chatInput.value = "";

    try {
      const response = await getChatResponse(userText);
      appendAssistantMessage(response);
    } catch (error) {
      appendErrorMessage("Oops! Something went wrong while retrieving the response. Please try again.");
    }
  };

  const getChatResponse = async (userText) => {
    const requestBody = {
      userText: userText,
    };
    const requestOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    };

    const response = await fetch(API_URL, requestOptions);
    if (!response.ok) {
      throw new Error("Request to /assist failed.");
    }

    const data = await response.json();
    return data.text;
  };

  const appendUserMessage = (message) => {
    const userMessage = createMessageElement("outgoing", message);
    chatContainer.appendChild(userMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  };

  const appendAssistantMessage = (message) => {
    const assistantMessage = createMessageElement("incoming", message);
    chatContainer.appendChild(assistantMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  };

  const appendErrorMessage = (message) => {
    const errorMessage = createMessageElement("error", message);
    chatContainer.appendChild(errorMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  };

  const createMessageElement = (className, content) => {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("chat", className);
    messageDiv.innerHTML = `<div class="chat-content">
          <div class="chat-details">
              <img src="${className === "outgoing" ? "https://w7.pngwing.com/pngs/81/570/png-transparent-profile-logo-computer-icons-user-user-blue-heroes-logo-thumbnail.png" : "https://img1.wsimg.com/isteam/ip/85d08d79-91db-410d-a442-2255f9b47c90/Aztra%20Bot%20Ver%203%20copy.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25"}" alt="${className}-img">
          <div style="flex-direction:column; width:100%; overlay-x: hidden;">${md.render(content)}</div>
          </div>
      </div>`;

    hljs.highlightAll();
    return messageDiv;
  };

  // Append the greeting message to the chat container
  appendAssistantMessage(greetingMessage);

  themeButton.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");
    localStorage.setItem("themeColor", themeButton.innerText);
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
  });

  deleteButton.addEventListener("click", () => {
    if (confirm("Are you sure you want to delete all the chats?")) {
      clearChat();
    }
  });

  const clearChat = () => {
    chatContainer.innerHTML = "";
    localStorage.removeItem("all-chats");
  };

  // Load saved theme
  const themeColor = localStorage.getItem("themeColor");
  if (themeColor) {
    document.body.classList.toggle("light-mode", themeColor === "light_mode");
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
  }
});



// const chatInput = document.querySelector("#chat-input");
// const sendButton = document.querySelector("#send-btn");
// const chatContainer = document.querySelector(".chat-container");
// const themeButton = document.querySelector("#theme-btn");
// const deleteButton = document.querySelector("#delete-btn");
// const menuIcon = document.querySelector("#menu-icon"); // Add menu icon reference
// const API_URL = "/assist";

// var md = window.markdownit();

// // Sidebar toggler
// const sidebar = document.querySelector('.sidebar');
// const toggleButton = document.querySelector('.toggle-sidebar-button');
// const hideSidebarButton = document.querySelector('.hide-sidebar-button');

// toggleButton.addEventListener('click', () => {
//   sidebar.classList.toggle('active');
// });

// hideSidebarButton.addEventListener('click', () => {
//   sidebar.classList.remove('active');
// });



// sendButton.addEventListener("click", () => {
//   console.log("Button clicked"); // Add this line
//   handleOutgoingChat();
// });

// chatInput.addEventListener("keydown", (e) => {
//   if (e.key === "Enter" && !e.shiftKey) {
//     e.preventDefault();
//     handleOutgoingChat();
//   }
// });

// const handleOutgoingChat = async () => {
//   const userText = chatInput.value.trim();
//   if (!userText) return;

//   appendUserMessage(userText);
//   chatInput.value = "";

//   try {
//     const response = await getChatResponse(userText);
//     appendAssistantMessage(response);
//   } catch (error) {
//     appendErrorMessage("Oops! Something went wrong while retrieving the response. Please try again.");
//   }
// };

// const getChatResponse = async (userText) => {
//   const requestBody = {
//     userText: userText,
//   };
//   const requestOptions = {
//     method: "POST",
//     headers: {
//       "Content-Type": "application/json",
//     },
//     body: JSON.stringify(requestBody),
//   };

//   const response = await fetch(API_URL, requestOptions);
//   if (!response.ok) {
//     throw new Error("Request to /assist failed.");
//   }

//   const data = await response.json();
//   return data.text;
// };

// const appendUserMessage = (message) => {
//   const userMessage = createMessageElement("outgoing", message);
//   chatContainer.appendChild(userMessage);
//   chatContainer.scrollTop = chatContainer.scrollHeight;
// };

// const appendAssistantMessage = (message) => {
//   const assistantMessage = createMessageElement("incoming", message);
//   chatContainer.appendChild(assistantMessage);
//   chatContainer.scrollTop = chatContainer.scrollHeight;
// };

// const appendErrorMessage = (message) => {
//   const errorMessage = createMessageElement("error", message);
//   chatContainer.appendChild(errorMessage);
//   chatContainer.scrollTop = chatContainer.scrollHeight;
// };

// const createMessageElement = (className, content) => {
//   const messageDiv = document.createElement("div");
//   messageDiv.classList.add("chat", className);
//   messageDiv.innerHTML = `<div class="chat-content">
//         <div class="chat-details">
//             <img src="${className === "outgoing" ? "user.jpg" : "https://img1.wsimg.com/isteam/ip/85d08d79-91db-410d-a442-2255f9b47c90/Aztra%20Bot%20Ver%203%20copy.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25"}" alt="${className}-img">
//         <div style="flex-direction:column; width:100%; overlay-x: hidden;">${md.render(content)}</div>
//         </div>
//     </div>`;

//   hljs.highlightAll();
//   return messageDiv;
// };

// themeButton.addEventListener("click", () => {
//   document.body.classList.toggle("light-mode");
//   localStorage.setItem("themeColor", themeButton.innerText);
//   themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
// });

// deleteButton.addEventListener("click", () => {
//   if (confirm("Are you sure you want to delete all the chats?")) {
//     clearChat();
//   }
// });

// const clearChat = () => {
//   chatContainer.innerHTML = "";
//   localStorage.removeItem("all-chats");
// };

// // Load saved theme
// const themeColor = localStorage.getItem("themeColor");
// if (themeColor) {
//   document.body.classList.toggle("light-mode", themeColor === "light_mode");
//   themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
// }