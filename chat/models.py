import uuid

from django.db import models
from django.db.models import Q


class Project(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="sessions",
    )
    name = models.CharField(max_length=255, blank=True, default="")
    config = models.JSONField(default=dict)
    active_branch_id = models.UUIDField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["project", "name"],
                condition=~Q(name=""),
                name="uniq_named_session_per_project",
            ),
        ]

    def __str__(self) -> str:
        label = self.name or str(self.id)
        return f"{self.project.name}:{label}"


class StoredMessage(models.Model):
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    position = models.PositiveIntegerField()
    message_type = models.CharField(max_length=32)
    payload = models.JSONField()

    class Meta:
        ordering = ["position"]
        constraints = [
            models.UniqueConstraint(
                fields=["session", "position"],
                name="uniq_message_position_per_session",
            ),
        ]


class BranchNodeRecord(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="branch_nodes",
    )
    parent = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="children_nodes",
    )
    compressed_payload = models.JSONField(null=True, blank=True)
    compressed_through = models.PositiveIntegerField()
    visible_end = models.PositiveIntegerField(null=True, blank=True)
    child_order = models.JSONField(default=list)

    class Meta:
        ordering = ["compressed_through"]
