// Initialize Lucide Icons
lucide.createIcons();

// Dynamic Text Flip/Move Up Effect for Hero Section Description
document.addEventListener('DOMContentLoaded', () => {
    const dynamicTextElement = document.getElementById('dynamic-text');
    if (!dynamicTextElement) {
        console.error("Element with ID 'dynamic-text' not found. Dynamic typing effect cannot be applied.");
        return;
    }

    const descriptions = ["Optimisation", "Data Science", "Language Models", "ML Research", "AI Safety", "LLM Reasoning"];
    let descriptionIndex = 0;
    const animationDuration = 700; // Match CSS animation duration in ms (for both flip-in and flip-out)
    const pauseDuration = 1500; // Pause after flip-in, before next flip-out starts

    function animateTextCycle() {
        // Step 1: Trigger flip-out animation
        dynamicTextElement.classList.remove('flip-in');
        dynamicTextElement.classList.add('flip-out');

        // Step 2: After flip-out animation completes, update content and trigger flip-in
        setTimeout(() => {
            // Move to the next description
            descriptionIndex = (descriptionIndex + 1) % descriptions.length;
            dynamicTextElement.textContent = descriptions[descriptionIndex];

            // Trigger flip-in animation
            dynamicTextElement.classList.remove('flip-out'); // Remove flip-out before adding flip-in
            dynamicTextElement.classList.add('flip-in');

            // Step 3: After flip-in animation completes, pause and then start the next cycle
            setTimeout(() => {
                animateTextCycle(); // Loop
            }, pauseDuration); // Pause after the flip-in
        }, animationDuration); // Wait for the flip-out animation to finish
    }

    // Initial setup: display the first text and immediately flip it in
    dynamicTextElement.textContent = descriptions[0];
    dynamicTextElement.classList.add('flip-in');

    // Start the continuous cycle after the initial flip-in completes and a pause
    // We need to wait for the first 'flip-in' animation to finish AND then the pauseDuration
    setTimeout(() => {
        animateTextCycle();
    }, animationDuration + pauseDuration);

});
