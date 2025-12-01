export default function DisclaimerBanner() {
    return (
        <div className="bg-red-600 text-white p-6 rounded-lg shadow-lg mb-8">
            <div className="flex items-start gap-4">
                <div className="text-4xl">⚠️</div>
                <div>
                    <h2 className="text-2xl font-bold mb-2">MEDICAL DISCLAIMER</h2>
                    <p className="text-lg mb-2">
                        <strong>This is an EXPERIMENTAL RESEARCH DEMONSTRATION ONLY.</strong>
                    </p>
                    <ul className="list-disc list-inside space-y-1 text-sm">
                        <li>This is <strong>NOT</strong> a medical device</li>
                        <li>This is <strong>NOT</strong> clinically validated</li>
                        <li>This is <strong>NOT</strong> FDA approved</li>
                        <li>This uses <strong>SYNTHETIC</strong> data only</li>
                        <li><strong>DO NOT</strong> use for medical decisions</li>
                        <li><strong>ALWAYS</strong> consult a healthcare professional</li>
                    </ul>
                </div>
            </div>
        </div>
    );
}
