using UnityEngine;
using System.IO;
using TMPro;  // For TextMeshPro (UI)
using System.Collections;

public class GeneralizedSimulationLogger : MonoBehaviour
{
    public float length = 5f; // Default length for torque calculation
    private Vector3 initialPosition;
    private float originalHeight;
    private float width;
    private float area;

    private StreamWriter writer;

    // Material Properties (Wood as the example material)
    private const float YoungsModulusWood = 1.0e10f;

    // For calculating averages
    private float totalStress = 0f;
    private float totalStrain = 0f;
    private float totalTorque = 0f;
    private float totalFrequency = 0f;
    private int simulationCount = 0;

    // Timer variables
    private float simulationTime = 0f;
    private const float maxSimulationTime = 5f; // Time in seconds for the simulation to run

    // Reference to the UI manager for updating results
    public SimulationUIManager uiManager;

    // Reference to the SendToFlask script for server communication
    private SendToFlask sendToFlask;

    void Start()
    {
        // Initialize the SendToFlask script
        sendToFlask = GetComponent<SendToFlask>();

        // Dynamically calculate model's physical properties
        Renderer renderer = GetComponent<Renderer>();
        if (renderer == null)
        {
            Debug.LogError("No Renderer component found on the model!");
            return;
        }

        originalHeight = renderer.bounds.size.y;
        width = renderer.bounds.size.x;
        area = renderer.bounds.size.x * renderer.bounds.size.z;

        if (renderer.bounds.size.z > 0)
        {
            length = renderer.bounds.size.z;
        }

        initialPosition = transform.position;

        // Prepare CSV file to store results
        string filePath = Application.dataPath + "/GeneralizedSimulationData.csv";
        writer = new StreamWriter(filePath, false);
        writer.WriteLine("HeightToWidthRatio,AvgStress,AvgStrain,AvgTorque,AvgFrequency");

        Debug.Log("Simulation started for model: " + gameObject.name);
    }

    void Update()
    {
        // Ensure the Rigidbody is disabled to prevent movement during simulations
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.isKinematic = true;
        }

        // Track simulation time
        simulationTime += Time.deltaTime;

        // Stop simulation after maxSimulationTime
        if (simulationTime >= maxSimulationTime)
        {
            // Calculate averages
            float avgStress = totalStress / simulationCount;
            float avgStrain = totalStrain / simulationCount;
            float avgTorque = totalTorque / simulationCount;
            float avgFrequency = totalFrequency / simulationCount;

            float heightToWidthRatio = originalHeight / width;

            // Log the result
            string result = $"HeightToWidthRatio={heightToWidthRatio}, Stress={avgStress}, Strain={avgStrain}, Torque={avgTorque}, Frequency={avgFrequency}";
            Debug.Log(result);

            // Update the UI with results
            if (uiManager != null)
            {
                uiManager.UpdateResults(result);
            }

            // Write to the CSV file
            writer.WriteLine($"{heightToWidthRatio},{avgStress},{avgStrain},{avgTorque},{avgFrequency}");

            // Send data to Flask server
            if (sendToFlask != null)
            {
                sendToFlask.SendData(heightToWidthRatio, avgStress, avgStrain, avgTorque, avgFrequency);
            }

            // Disable further updates
            enabled = false;
        }
        else
        {
            // Calculate physical properties during the simulation
            float force = Mathf.Abs(GetComponent<Rigidbody>().mass * Physics.gravity.y);
            float stress = force / area;
            float strain = stress / YoungsModulusWood;
            totalStrain += strain;

            float torque = force * length;
            totalTorque += torque;

            float frequency = Mathf.Abs(Mathf.Sin(Time.time));
            totalFrequency += frequency;

            totalStress += stress;
            simulationCount++;
        }
    }

    void OnDestroy()
    {
        if (writer != null)
        {
            writer.Close();
        }
    }
}
