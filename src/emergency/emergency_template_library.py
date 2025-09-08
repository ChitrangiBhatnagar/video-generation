import os
import json
import yaml
import logging
import shutil
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EmergencyTemplate:
    """Template for emergency video generation."""
    # Template identification
    id: str
    name: str
    version: str
    emergency_type: str
    priority_level: str
    
    # Content
    prompt_template: str
    visual_elements: List[Dict[str, Any]] = field(default_factory=list)
    audio_elements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Accessibility
    caption_template: Optional[str] = None
    alt_text_template: Optional[str] = None
    high_contrast_mode: bool = True
    
    # Compliance
    required_disclaimers: List[str] = field(default_factory=list)
    jurisdiction_requirements: Dict[str, List[str]] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    author: Optional[str] = None
    
    def format_prompt(self, variables: Dict[str, str]) -> str:
        """Format the prompt template with the given variables.
        
        Args:
            variables: Dictionary of variable names and values.
            
        Returns:
            Formatted prompt string.
        """
        prompt = self.prompt_template
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            prompt = prompt.replace(placeholder, value)
        return prompt
    
    def format_captions(self, variables: Dict[str, str]) -> Optional[str]:
        """Format the caption template with the given variables.
        
        Args:
            variables: Dictionary of variable names and values.
            
        Returns:
            Formatted caption string or None if no caption template.
        """
        if not self.caption_template:
            return None
            
        captions = self.caption_template
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            captions = captions.replace(placeholder, value)
        return captions
    
    def get_disclaimers(self, jurisdiction: Optional[str] = None) -> List[str]:
        """Get required disclaimers for this template.
        
        Args:
            jurisdiction: Optional jurisdiction code.
            
        Returns:
            List of required disclaimer strings.
        """
        disclaimers = self.required_disclaimers.copy()
        
        # Add jurisdiction-specific disclaimers if applicable
        if jurisdiction and jurisdiction in self.jurisdiction_requirements:
            disclaimers.extend(self.jurisdiction_requirements[jurisdiction])
            
        return disclaimers


class EmergencyTemplateLibrary:
    """Library for managing emergency video generation templates."""
    
    def __init__(self, template_directory: str = "data/emergency_templates"):
        """Initialize the emergency template library.
        
        Args:
            template_directory: Directory containing template files.
        """
        self.template_directory = template_directory
        self.templates: Dict[str, EmergencyTemplate] = {}
        
        # Create template directory if it doesn't exist
        os.makedirs(template_directory, exist_ok=True)
        
        # Load templates
        self._load_templates()
        
        logger.info(f"Emergency Template Library initialized with {len(self.templates)} templates")
    
    def _load_templates(self):
        """Load templates from the template directory."""
        if not os.path.exists(self.template_directory):
            logger.warning(f"Template directory {self.template_directory} does not exist")
            return
        
        try:
            # Get all JSON and YAML files in the template directory
            template_files = [f for f in os.listdir(self.template_directory) 
                             if f.endswith('.json') or f.endswith('.yaml') or f.endswith('.yml')]
            
            for filename in template_files:
                try:
                    file_path = os.path.join(self.template_directory, filename)
                    
                    # Load template data based on file extension
                    if filename.endswith('.json'):
                        with open(file_path, 'r') as f:
                            template_data = json.load(f)
                    else:  # YAML file
                        with open(file_path, 'r') as f:
                            template_data = yaml.safe_load(f)
                    
                    # Create template object
                    template = EmergencyTemplate(**template_data)
                    self.templates[template.id] = template
                    
                except Exception as e:
                    logger.error(f"Error loading template from {filename}: {e}")
            
            logger.info(f"Loaded {len(self.templates)} templates from {self.template_directory}")
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    def get_template(self, template_id: str) -> Optional[EmergencyTemplate]:
        """Get a template by ID.
        
        Args:
            template_id: ID of the template to retrieve.
            
        Returns:
            EmergencyTemplate object or None if not found.
        """
        return self.templates.get(template_id)
    
    def find_templates(self, 
                      emergency_type: Optional[str] = None, 
                      priority_level: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> List[EmergencyTemplate]:
        """Find templates matching the given criteria.
        
        Args:
            emergency_type: Type of emergency situation.
            priority_level: Priority level for the emergency.
            tags: List of tags to match.
            
        Returns:
            List of matching EmergencyTemplate objects.
        """
        results = []
        
        for template in self.templates.values():
            # Check emergency type
            if emergency_type and template.emergency_type != emergency_type:
                continue
                
            # Check priority level
            if priority_level and template.priority_level != priority_level:
                continue
                
            # Check tags
            if tags:
                if not all(tag in template.tags for tag in tags):
                    continue
            
            results.append(template)
        
        # Sort results by priority level if available
        if results and hasattr(results[0], 'priority_level'):
            results.sort(key=lambda t: t.priority_level, reverse=True)
        
        return results
    
    def add_template(self, template: EmergencyTemplate) -> bool:
        """Add a new template to the library.
        
        Args:
            template: EmergencyTemplate object to add.
            
        Returns:
            bool: True if the template was added successfully.
        """
        try:
            # Check if template with this ID already exists
            if template.id in self.templates:
                logger.warning(f"Template with ID {template.id} already exists")
                return False
            
            # Add to in-memory collection
            self.templates[template.id] = template
            
            # Save to file
            file_path = os.path.join(self.template_directory, f"{template.id}.json")
            with open(file_path, 'w') as f:
                # Convert dataclass to dictionary
                template_dict = {k: v for k, v in template.__dict__.items()}
                json.dump(template_dict, f, indent=2)
            
            logger.info(f"Added template {template.id} to library")
            return True
        except Exception as e:
            logger.error(f"Error adding template: {e}")
            return False
    
    def update_template(self, template: EmergencyTemplate) -> bool:
        """Update an existing template in the library.
        
        Args:
            template: EmergencyTemplate object with updated data.
            
        Returns:
            bool: True if the template was updated successfully.
        """
        try:
            # Check if template exists
            if template.id not in self.templates:
                logger.warning(f"Template with ID {template.id} does not exist")
                return False
            
            # Update in-memory collection
            self.templates[template.id] = template
            
            # Save to file
            file_path = os.path.join(self.template_directory, f"{template.id}.json")
            with open(file_path, 'w') as f:
                # Convert dataclass to dictionary
                template_dict = {k: v for k, v in template.__dict__.items()}
                json.dump(template_dict, f, indent=2)
            
            logger.info(f"Updated template {template.id} in library")
            return True
        except Exception as e:
            logger.error(f"Error updating template: {e}")
            return False
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a template from the library.
        
        Args:
            template_id: ID of the template to delete.
            
        Returns:
            bool: True if the template was deleted successfully.
        """
        try:
            # Check if template exists
            if template_id not in self.templates:
                logger.warning(f"Template with ID {template_id} does not exist")
                return False
            
            # Remove from in-memory collection
            del self.templates[template_id]
            
            # Remove file
            file_path = os.path.join(self.template_directory, f"{template_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logger.info(f"Deleted template {template_id} from library")
            return True
        except Exception as e:
            logger.error(f"Error deleting template: {e}")
            return False
    
    def export_template(self, template_id: str, export_path: str) -> bool:
        """Export a template to a file.
        
        Args:
            template_id: ID of the template to export.
            export_path: Path to export the template to.
            
        Returns:
            bool: True if the template was exported successfully.
        """
        try:
            # Check if template exists
            if template_id not in self.templates:
                logger.warning(f"Template with ID {template_id} does not exist")
                return False
            
            # Get template
            template = self.templates[template_id]
            
            # Determine format based on export path
            if export_path.endswith('.json'):
                # Export as JSON
                with open(export_path, 'w') as f:
                    # Convert dataclass to dictionary
                    template_dict = {k: v for k, v in template.__dict__.items()}
                    json.dump(template_dict, f, indent=2)
            elif export_path.endswith('.yaml') or export_path.endswith('.yml'):
                # Export as YAML
                with open(export_path, 'w') as f:
                    # Convert dataclass to dictionary
                    template_dict = {k: v for k, v in template.__dict__.items()}
                    yaml.dump(template_dict, f)
            else:
                logger.error(f"Unsupported export format for {export_path}")
                return False
            
            logger.info(f"Exported template {template_id} to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting template: {e}")
            return False
    
    def import_template(self, import_path: str) -> Optional[str]:
        """Import a template from a file.
        
        Args:
            import_path: Path to import the template from.
            
        Returns:
            str: ID of the imported template or None if import failed.
        """
        try:
            # Load template data based on file extension
            if import_path.endswith('.json'):
                with open(import_path, 'r') as f:
                    template_data = json.load(f)
            elif import_path.endswith('.yaml') or import_path.endswith('.yml'):
                with open(import_path, 'r') as f:
                    template_data = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported import format for {import_path}")
                return None
            
            # Create template object
            template = EmergencyTemplate(**template_data)
            
            # Add to library
            success = self.add_template(template)
            if not success:
                return None
            
            logger.info(f"Imported template {template.id} from {import_path}")
            return template.id
        except Exception as e:
            logger.error(f"Error importing template: {e}")
            return None
    
    def get_all_templates(self) -> List[EmergencyTemplate]:
        """Get all templates in the library.
        
        Returns:
            List of all EmergencyTemplate objects.
        """
        return list(self.templates.values())
    
    def backup_templates(self, backup_dir: str) -> bool:
        """Backup all templates to a directory.
        
        Args:
            backup_dir: Directory to backup templates to.
            
        Returns:
            bool: True if backup was successful.
        """
        try:
            # Create backup directory if it doesn't exist
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy all template files to backup directory
            for filename in os.listdir(self.template_directory):
                if filename.endswith('.json') or filename.endswith('.yaml') or filename.endswith('.yml'):
                    src_path = os.path.join(self.template_directory, filename)
                    dst_path = os.path.join(backup_dir, filename)
                    shutil.copy2(src_path, dst_path)
            
            logger.info(f"Backed up {len(self.templates)} templates to {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"Error backing up templates: {e}")
            return False