// Initialize Lucide Icons
// This script ensures all `data-lucide` attributes are converted to SVG icons.
// It should be placed at the end of the body or after the content where icons are used.
lucide.createIcons();

// Dynamic Typing Effect for Hero Section Description
document.addEventListener('DOMContentLoaded', () => {
    const dynamicTextElement = document.getElementById('dynamic-text');
    // Ensure the element exists before trying to manipulate it
    if (!dynamicTextElement) {
        console.error("Element with ID 'dynamic-text' not found. Dynamic typing effect cannot be applied.");
        return;
    }

    const descriptions = ["Machine Learning Engineer", "ML Researcher", "LLM Specialist", "Data Scientist"];
    let descriptionIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    let typingSpeed = 100; // milliseconds per character
    let deletingSpeed = 50; // milliseconds per character
    let pauseBeforeNext = 1500; // milliseconds to pause before typing next description

    function typeEffect() {
        const currentDescription = descriptions[descriptionIndex];
        if (isDeleting) {
            // Deleting text
            dynamicTextElement.textContent = currentDescription.substring(0, charIndex - 1);
            charIndex--;
            if (charIndex === 0) {
                isDeleting = false;
                descriptionIndex = (descriptionIndex + 1) % descriptions.length; // Move to next description
                setTimeout(typeEffect, 500); // Pause before typing next
            } else {
                setTimeout(typeEffect, deletingSpeed);
            }
        } else {
            // Typing text
            dynamicTextElement.textContent = currentDescription.substring(0, charIndex + 1);
            charIndex++;
            if (charIndex === currentDescription.length) {
                isDeleting = true;
                setTimeout(typeEffect, pauseBeforeNext); // Pause before deleting
            } else {
                setTimeout(typeEffect, typingSpeed);
            }
        }
    }

    // Start the typing effect when the DOM is fully loaded
    typeEffect();
});
