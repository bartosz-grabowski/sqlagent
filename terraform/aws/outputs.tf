output "application_url" {
  description = "Public URL of the SQLAgent API."
  value       = "http://${aws_instance.sqlagent.public_ip}:${var.app_port}"
}

output "instance_id" {
  description = "EC2 instance ID hosting SQLAgent."
  value       = aws_instance.sqlagent.id
}

output "instance_public_ip" {
  description = "Public IPv4 address of the SQLAgent EC2 instance."
  value       = aws_instance.sqlagent.public_ip
}

output "ssm_start_session_command" {
  description = "Command that opens a Session Manager shell on the instance."
  value       = "aws ssm start-session --target ${aws_instance.sqlagent.id}"
}

output "resource_prefix" {
  description = "Generated prefix used for AWS resources in this deployment."
  value       = local.resource_name
}
