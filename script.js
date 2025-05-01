function checkEmail() {
    // Get the email text from the textarea
    var emailText = document.getElementById('email-text').value;

    // Check if there's input
    if (emailText.trim() === "") {
        alert("Please enter an email text!");
        return;
    }

    // Send the text to the Flask backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email_text: emailText }),
    })
    .then(response => response.json())
    .then(data => {
        // Show the result
        document.getElementById('result').textContent = "Result: " + data.result;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
