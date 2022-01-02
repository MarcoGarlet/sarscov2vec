from ProjectExtractorFold.TextExtractor import TextExtractor
from ProjectExtractorFold.CEWordExtractionProject import CEWordExtractionProject

class TextExtractorProject(TextExtractor):
	def __init__(self, handle_file_list):
		self.handle_list = handle_file_list
		self.myCEWordExtractionAlg=CEWordExtractionProject(self.handle_list)