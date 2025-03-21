from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

def setup_telemetry(service_name="research-ai"):
    """Set up OpenTelemetry tracing and metrics for SignOz monitoring"""
    resource = Resource(attributes={
        SERVICE_NAME: service_name
    })
    
    # Set up tracing
    tracer_provider = TracerProvider(resource=resource)
    span_processor = BatchSpanProcessor(OTLPSpanExporter())
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)
    
    # Set up metrics
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter()
    )
    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=[metric_reader])
    )
    
    return trace.get_tracer(service_name), metrics.get_meter(service_name)
