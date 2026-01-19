document.addEventListener("DOMContentLoaded", function () {
    // Lista de imágenes disponibles
    const images = [
        "./js/assets/1.jpg",
        "./js/assets/2.jpg",
        "./js/assets/3.jpg",
        "./js/assets/4.jpg"
    ];

    let currentIndex = 0;

    // Obtener referencia a la imagen, botón y loading
    const imageElement = document.getElementById("mriImage");
    const nextButton = document.getElementById("nextButton");
    const loadingElement = document.getElementById("loading");

    // Verificar si los elementos existen
    if (!imageElement || !nextButton || !loadingElement) {
        console.error("No se encontraron los elementos.");
        return;
    }

    console.log("Imagen, botón y loading encontrados correctamente.");

    // Función para mostrar loading y cambiar imagen
    function nextImage() {
        // Mostrar el loading y ocultar la imagen
        loadingElement.classList.remove("hidden");
        imageElement.classList.add("hidden");
        nextButton.disabled = true; // Deshabilitar el botón mientras carga

        setTimeout(() => {
            // Cambiar la imagen después de 1.5 segundos
            currentIndex = (currentIndex + 1) % images.length;
            imageElement.src = images[currentIndex];

            console.log("Cambiando imagen a:", imageElement.src);

            // Ocultar el loading y mostrar la imagen
            loadingElement.classList.add("hidden");
            imageElement.classList.remove("hidden");
            nextButton.disabled = false; // Habilitar el botón nuevamente
        }, 2000);
    }

    // Agregar evento al botón
    nextButton.addEventListener("click", nextImage);
});
