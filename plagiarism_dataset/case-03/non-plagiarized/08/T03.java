import java.util.Scanner;
public class T03 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter weight in pounds: ");
        double weight = sc.nextDouble();
        System.out.print("Enter feet: ");
        double feet = sc.nextDouble();
        System.out.print("Enter Inches: ");
        double inches = sc.nextDouble();
        double height = feet * 12 + inches;
        double bmi = weight * 0.45359237 / Math.pow((height * 0.0254), 2);
        System.out.println("BMI is "+ bmi);
        if(bmi<18.5){
            System.out.println("underweight");
        } else if(bmi<25){
            System.out.println("normal");
        } else if(bmi<35){
            System.out.println("overweight");
        } else{
            System.out.println("obese");
        }
    }
}
