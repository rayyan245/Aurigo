using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class SendToFlask : MonoBehaviour
{
    // Flask server URL
    private string serverUrl = "http://127.0.0.1:5000/log";

    /// <summary>
    /// Sends simulation data to the Flask server.
    /// </summary>
    /// <param name="heightToWidthRatio">The height-to-width ratio of the object.</param>
    /// <param name="avgStress">The average stress value.</param>
    /// <param name="avgStrain">The average strain value.</param>
    /// <param name="avgTorque">The average torque value.</param>
    /// <param name="avgFrequency">The average frequency value.</param>
    public void SendData(float heightToWidthRatio, float avgStress, float avgStrain, float avgTorque, float avgFrequency)
    {
        // Create a JSON object with the data
        var jsonData = JsonUtility.ToJson(new
        {
            HeightToWidthRatio = heightToWidthRatio,
            AvgStress = avgStress,
            AvgStrain = avgStrain,
            AvgTorque = avgTorque,
            AvgFrequency = avgFrequency
        });

        // Start the coroutine to send the POST request
        StartCoroutine(PostRequest(serverUrl, jsonData));
    }

    /// <summary>
    /// Sends a POST request with JSON data to the server.
    /// </summary>
    /// <param name="url">The server URL.</param>
    /// <param name="jsonData">The JSON data to send.</param>
    /// <returns></returns>
    private IEnumerator PostRequest(string url, string jsonData)
    {
        UnityWebRequest request = new UnityWebRequest(url, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        // Check the result
        if (request.result == UnityWebRequest.Result.Success)
        {
            Debug.Log("Data sent successfully: " + request.downloadHandler.text);
        }
        else
        {
            Debug.LogError("Failed to send data: " + request.error);
        }
    }
}
