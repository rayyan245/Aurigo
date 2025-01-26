using UnityEngine;
using TMPro;  // For TextMeshPro
using UnityEngine.UI;  // For Button

public class SimulationUIManager : MonoBehaviour
{
    public Button startButton;  // Reference to the Start Button
    public TextMeshProUGUI resultText;  // Reference to the TextMeshPro for displaying results
    public GeneralizedSimulationLogger simulationScript;  // Reference to the simulation script

    void Start()
    {
        // Ensure the start button triggers the StartSimulation method when clicked
        startButton.onClick.AddListener(StartSimulation);
        
        // Set initial text to show the user
        resultText.text = "Waiting for results...";
    }

    // This method starts the simulation when the start button is clicked
    void StartSimulation()
    {
        resultText.text = "Running simulation...";  // Update the result text to show that the simulation is running
        simulationScript.enabled = true;  // Enable the simulation script to start the simulation
    }

    // This method updates the result text once the simulation finishes
public void UpdateResults(string results)
{
    Debug.Log("Updating result text: " + results);  // Debug log to check if this is being called
    resultText.text = results;
}

}
