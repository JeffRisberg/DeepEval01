
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import SummarizationMetric

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)

contextual_precision_metric = ContextualPrecisionMetric(
    threshold=0.7,
    model="gpt-4o-mini",
    include_reason=True
)

test_case = LLMTestCase(
    input="Do I qualify for remote work?",
    actual_output="A clean desk is a requirement and good performance may be a prerequisite for remote work.",
    expected_output="Good performance is a prerequisite for remote work", # needed for contextual precision metric
    retrieval_context=[
        "Determining if you qualify for remote work involves several factors. Here’s how to assess your eligibility:",
        "1. Review Company Policy:",
        "Familiarize yourself with ComputeCore Systems’ remote work policy, which can be found in the employee handbook",
        "3. Performance Evaluation",
        "Good performance in your current role is often a prerequisite for remote work",
        "4. Discuss with Your Supervisor:"
        "Initiate a conversation with your supervisor to express your interest in remote work. ",
        "Be prepared to discuss how you can maintain productivity while working remotely."
    ]
)

print(f"Expected output: '{test_case.expected_output}'")
print(f"Actual output: '{test_case.actual_output}'")

answer_relevancy_metric.measure(test_case)
print(f"Answer relevancy score: {answer_relevancy_metric.score}")
print(f"Answer relevancy reason: {answer_relevancy_metric.reason}")
print()

contextual_precision_metric.measure(test_case)
#print(contextual_precision_metric.score)
#print(contextual_precision_metric.reason)
print(f"Contextual Precision score: {contextual_precision_metric.score}")
print(f"Contextual Precision reason: {contextual_precision_metric.reason}")
print()

summarization_metric = SummarizationMetric(
    threshold=0.5,
    model="gpt-4o-mini",
)

summarization_test_case = LLMTestCase(
    input="The process for requesting remote work is to read the employee handbook, contact your manager, and depends upon good performance",
    actual_output="To request remote work, contact your manager, and have good performance"
)

print(f"Input: '{summarization_test_case.input}'")
print(f"Output: '{summarization_test_case.actual_output}'")

summarization_metric.measure(summarization_test_case)
#print(summarization_metric.score)
#print(summarization_metric.reason)
print(f"Summarization score: {summarization_metric.score}")
print(f"Summarization reason: {summarization_metric.reason}")
print()
