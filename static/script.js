function validateSignupForm() {
    const nameInput = document.querySelector("#name");
    const emailInput = document.querySelector("#email");
    const passwordInput = document.querySelector("#password");
  
    if (nameInput.value === "") {
      alert("Please enter your name.");
      return false;
    }
  
    if (emailInput.value === "") {
      alert("Please enter your email address.");
      return false;
    }
  
    if (passwordInput.value === "") {
      alert("Please enter a password.");
      return false;
    }
  
    return true;
  }
  